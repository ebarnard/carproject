#![allow(non_snake_case)]

extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate toml;

extern crate control_model;
extern crate controller;
extern crate estimator;
extern crate prelude;
extern crate track;
extern crate ui;

mod config;
pub mod visualisation;

use nalgebra::{self, MatrixMN, Vector3};
use std::sync::Arc;
use std::time::{Duration, Instant};

use prelude::*;
use controller::Controller;
use control_model::ControlModel;
use estimator::{Estimator, JointEKF};
use track::TrackAndLookup;

use config::{CarConfig, ControllerConfig};

pub use control_model::{Control, State};
pub use estimator::Measurement;

pub trait ControllerEstimator: Send {
    fn optimise_dt(&self) -> float;
    fn horizon_dt(&self) -> float;
    fn N(&self) -> u32;
    fn np(&self) -> u32;
    fn track(&self) -> &Arc<TrackAndLookup>;

    fn control_applied(&mut self, control: Control, control_time: Duration);

    fn measurement(&mut self, measurement: Option<Measurement>, measurement_time: Duration);

    fn step(&mut self, control_application_time: Duration) -> StepResult;
}

pub struct StepResult<'a> {
    pub current_state: State,
    pub control_application_state: State,
    pub control_horizon: &'a [(Control, State)],
    pub params: &'a [float],
}

pub struct ControllerEstimatorImpl<M: ControlModel, C: Controller<M>>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    config: CarConfig,
    track: Arc<TrackAndLookup>,
    model: M,
    controller: C,
    estimator: JointEKF<M>,
    u: Matrix<M::NI, Dy>,
    horizon: Vec<(Control, State)>,
    /// The time at which the first input of `u` will be applied
    u_target_time: Duration,
    /// The control signal currently being applied.
    current_control: Vector<M::NI>,
    /// The current time of the estimator.
    estimator_time: Duration,
}

// TODO: Remove this hacky impl once nalgebra uses const generics.
unsafe impl<M: ControlModel, C: Controller<M>> Send for ControllerEstimatorImpl<M, C>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
}

pub fn controllers_from_config() -> (Arc<TrackAndLookup>, Vec<Box<ControllerEstimator>>) {
    let config = ControllerConfig::load();
    let track = track::Track::load(&*config.track);
    // TODO: Init w & h from camera
    let track = Arc::new(TrackAndLookup::new(track, 1280 * 2, 1024 * 2));
    let R = Vector3::from_column_slice(&config.R);

    let car_controllers = config
        .cars
        .into_iter()
        .map(|car| {
            CONTROLLERS
                .iter()
                .find(|c| c.0() == car.model && c.1() == car.controller)
                .expect("controller not found")
                .2(car, track.clone(), &R)
        })
        .collect();

    (track, car_controllers)
}

fn new<M: 'static + ControlModel, C: 'static + Controller<M>>(
    config: CarConfig,
    track: Arc<TrackAndLookup>,
    R: &Vector3<float>,
) -> Box<ControllerEstimator>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    let model = M::new();
    let N = config.N;
    let optimise_dt = config.optimise_dt;

    let mut controller = C::new(&model, N, &track);
    let u_min = Vector::<M::NI>::from_column_slice(&config.u_min);
    let u_max = Vector::<M::NI>::from_column_slice(&config.u_max);
    controller.update_input_bounds(u_min, u_max);

    let initial_params = Vector::<M::NP>::from_column_slice(&config.initial_params);
    let Q_state = Matrix::from_diagonal(&Vector::<M::NS>::from_column_slice(&config.Q_state));
    let Q_initial_params = Matrix::from_diagonal(&Vector::<M::NP>::from_column_slice(
        &config.Q_initial_params,
    ));
    let Q_params = &Q_initial_params * config.Q_params_multiplier;
    let R = Matrix::from_diagonal(R);
    let estimator = JointEKF::<M>::new(initial_params, Q_state, Q_params, Q_initial_params, R);

    let mut u = MatrixMN::zeros_generic(<M::NI as DimName>::name(), Dy::new(N as usize + 1));
    // Start with non-zero inputs to avoid strange initial solve behaviour.
    u.row_mut(0).fill(config.u_max[0] * 1e-2);
    for i in 0..u.shape().1 {
        u[(1, i)] = 1e-2 * if i % 2 == 0 {
            config.u_min[1]
        } else {
            config.u_max[1]
        };
    }

    let mut controller = Box::new(ControllerEstimatorImpl {
        config,
        track,
        model,
        controller,
        estimator,
        u,
        horizon: vec![Default::default(); N as usize],
        u_target_time: Duration::from_secs(0),
        current_control: Vector::<M::NI>::zeros(),
        estimator_time: Duration::from_secs(0),
    });

    // Run the controller a few time to initialise rho.
    for _ in 0..100 {
        controller.step(secs_to_duration(optimise_dt));
    }

    controller
}

impl<M: 'static + ControlModel, C: Controller<M>> ControllerEstimator
    for ControllerEstimatorImpl<M, C>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn optimise_dt(&self) -> float {
        self.config.optimise_dt
    }

    fn horizon_dt(&self) -> float {
        self.config.horizon_dt
    }

    fn N(&self) -> u32 {
        self.config.N
    }

    fn np(&self) -> u32 {
        M::NP::dim() as u32
    }

    fn track(&self) -> &Arc<TrackAndLookup> {
        &self.track
    }

    // `control_time` is the absolute time at which the control input was applied.
    fn control_applied(&mut self, control: Control, control_time: Duration) {
        // Update estimator based on previous control value and time
        // Save current control and time
        let current_control_applied_duration = control_time
            .checked_sub(self.estimator_time)
            .expect("control must be applied after the previous control or measurement");

        let (_, _) = self.estimator.step(
            &mut self.model,
            duration_to_secs(current_control_applied_duration),
            &self.current_control,
            None,
        );

        self.current_control = self.model.u_from_control(&control);
        self.estimator_time = control_time;
    }

    // `measurement_time` is the absolute time at which the measurement was taken.
    fn measurement(&mut self, measurement: Option<Measurement>, measurement_time: Duration) {
        // Update estimator based on previous control value and time
        // Save current control and time
        let current_control_applied_duration = measurement_time
            .checked_sub(self.estimator_time)
            .expect("measurement must be applied after the previous control or measurement");

        let (_, _) = self.estimator.step(
            &mut self.model,
            duration_to_secs(current_control_applied_duration),
            &self.current_control,
            measurement,
        );

        self.estimator_time = measurement_time;
    }

    // `control_application_time` is the time at which the returned control input will be applied.
    fn step(&mut self, control_application_time: Duration) -> StepResult {
        let step_start = Instant::now();

        // Calculate how far in the future the control will be applied.
        let control_application_delay = control_application_time
            .checked_sub(self.estimator_time)
            .expect("target control application must be after the current control or measurement");

        self.u_target_time = control_application_time;

        // Get current predicted state and parameters
        let (predicted_state, predicted_params) =
            self.estimator
                .step(&mut self.model, 0.0, &self.current_control, None);

        // Advance inputs so the first column of self.u contains the previously applied input.
        // TODO: Actually advance the values in the input matrix.

        // Simulate state evolution so `control_application_state` contains the state of the car
        // when the control input will be applied.
        let control_application_state = self.model.step(
            duration_to_secs(control_application_delay),
            &predicted_state,
            &self.u.column(0).into_owned(),
            &predicted_params,
        );

        // Optimise control inputs.
        let controller_time_limit = control_application_delay
            .checked_sub(step_start.elapsed())
            .unwrap_or_default();
        let (horizon_ctrl, horizon_state) = self.controller.step(
            &mut self.model,
            self.config.horizon_dt,
            &control_application_state,
            &self.u,
            &predicted_params,
            controller_time_limit,
        );
        self.u
            .columns_mut(0, self.config.N as usize)
            .copy_from(horizon_ctrl);

        for i in 0..(self.config.N as usize) {
            let control = self.model
                .u_to_control(&horizon_ctrl.column(i).into_owned());
            let state = self.model.x_to_state(&horizon_state.column(i).into_owned());
            self.horizon[i] = (control, state);
        }

        StepResult {
            current_state: self.model.x_to_state(&predicted_state),
            control_application_state: self.model.x_to_state(&control_application_state),
            control_horizon: &self.horizon,
            params: predicted_params.as_slice(),
        }
    }
}

use control_model::*;

macro_rules! as_expr {
    ($e:expr) => {$e};
}

macro_rules! expand_controllers {
    ($models:tt; ($($ex:tt)*);) => {
        as_expr!(&[$($ex,)*])
    };

    (($($model:ident,)*); ($($ex:tt)*); $controller:ident, $($tail:tt)*) => {
        expand_controllers!(
            ($($model,)*);
            ($($ex)* $((
                control_model::$model::name,
                controller::$controller::<$model>::name,
                new::<control_model::$model, controller::$controller<control_model::$model>>
            ))*);
            $($tail)*
        )
    };
}

macro_rules! controllers {
    (
        models {
            $($model:ident,)*
        }
        controllers {
            $($controllers:ident,)*
        }
    ) => {
        static CONTROLLERS: &'static [
            (
                fn() -> &'static str,
                fn() -> &'static str,
                fn(CarConfig, Arc<TrackAndLookup>, &Vector3<float>) -> Box<ControllerEstimator>
            )
        ] = expand_controllers!(($($model,)*); (); $($controllers,)*);
    };
}

controllers! {
    models {
        DirectVelocity,
        NoSlipPoint,
    }

    controllers {
        MpcDistance,
        MpcPosition,
    }
}
