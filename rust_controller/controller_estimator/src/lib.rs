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

use nalgebra::{self, DimAdd, DimSum, MatrixMN, U0, U3, Vector3};
use std::sync::Arc;
use std::time::Duration;

use prelude::*;
use controller::Controller;
use control_model::ControlModel;
use estimator::{Estimator, JointEKF};
use track::Track;

use config::ControllerConfig;

pub use control_model::{Control, State};
pub use estimator::Measurement;

pub trait ControllerEstimator {
    fn optimise_dt(&self) -> float;
    fn horizon_dt(&self) -> float;
    fn N(&self) -> u32;
    fn np(&self) -> u32;
    fn track(&self) -> &Arc<Track>;

    fn step(
        &mut self,
        measurement: Option<Measurement>,
        measurement_time: Duration,
        control_time: Duration,
    ) -> StepResult;
}

pub struct StepResult<'a> {
    pub predicted_state: State,
    pub horizon: &'a [(Control, State)],
    pub params: &'a [float],
}

pub struct ControllerEstimatorImpl<M: ControlModel, C: Controller<M>>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<U3, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    config: config::ControllerConfig,
    track: Arc<Track>,
    model: M,
    controller: C,
    estimator: JointEKF<M>,
    params: Vector<M::NP>,
    u: Matrix<M::NI, Dy>,
    horizon: Vec<(Control, State)>,
    prev_measurement_time: Duration,
    prev_control_time: Duration,
}

pub fn controller_from_config() -> Box<ControllerEstimator> {
    let config = config::ControllerConfig::load();
    let track = Arc::new(track::Track::load(&*config.track));
    CONTROLLERS
        .iter()
        .find(|c| c.0() == config.model && c.1() == config.controller)
        .expect("controller not found")
        .2(config, track)
}

fn new<M: 'static + ControlModel, C: 'static + Controller<M>>(
    config: ControllerConfig,
    track: Arc<Track>,
) -> Box<ControllerEstimator>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<U3, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    let model = M::new();
    let N = config.N;

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
    let R = Matrix::from_diagonal(&Vector3::from_column_slice(&config.R));

    let state_estimator = JointEKF::<M>::new(Q_state, Q_params, Q_initial_params, R);

    Box::new(ControllerEstimatorImpl {
        config,
        track,
        model,
        controller: controller,
        estimator: state_estimator,
        params: initial_params,
        u: MatrixMN::zeros_generic(<M::NI as DimName>::name(), Dy::new(N as usize + 1)),
        horizon: vec![Default::default(); N as usize],
        prev_measurement_time: Duration::from_secs(0),
        prev_control_time: Duration::from_secs(0),
    })
}

impl<M: 'static + ControlModel, C: Controller<M>> ControllerEstimator
    for ControllerEstimatorImpl<M, C>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<U3, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
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

    fn track(&self) -> &Arc<Track> {
        &self.track
    }

    // `measurement_time` is the absolute time at which the measurement was taken
    // `control_time` is the absolute time at which the returned control input will be applied
    fn step(
        &mut self,
        measurement: Option<Measurement>,
        measurement_time: Duration,
        control_time: Duration,
    ) -> StepResult {
        // Estimate state and params.
        // TODO: Support accepting measurements even if the controller is delayed.
        let measurement_delay = measurement_time
            .checked_sub(self.prev_measurement_time)
            .expect("measurements must be applied in temporal order");
        let (predicted_state, p) = self.estimator.step(
            &mut self.model,
            duration_to_secs(measurement_delay),
            &self.u.column(0).into_owned(),
            measurement,
            &self.params,
        );
        self.params = p;

        // Advance inputs so the first column of self.u contains the previously applied input.
        // TODO: Actually advance the values in the input matrix.
        let _control_delay = control_time
            .checked_sub(self.prev_control_time)
            .expect("control must be applied after the previous control");

        // Simulate state evolution so `control_application_state` contains the state of the car
        // when the control input will be applied.
        let measurement_control_delay = control_time
            .checked_sub(measurement_time)
            .expect("the control cannot be applied before the measurement");
        let control_application_state = self.model.step(
            duration_to_secs(measurement_control_delay),
            &predicted_state,
            &self.u.column(0).into_owned(),
            &self.params,
        );

        // Optimise control inputs.
        let (horizon_ctrl, horizon_state) = self.controller.step(
            &mut self.model,
            self.config.horizon_dt,
            &control_application_state,
            &self.u,
            &self.params,
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

        // Update measurement and control application timestamps.
        self.prev_control_time = control_time;
        self.prev_measurement_time = measurement_time;

        StepResult {
            predicted_state: self.model.x_to_state(&predicted_state),
            horizon: &self.horizon,
            params: &self.params.as_slice(),
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
                fn(ControllerConfig, Arc<Track>) -> Box<ControllerEstimator>
            )
        ] = expand_controllers!(($($model,)*); (); $($controllers,)*);
    };
}

controllers! {
    models {
        DirectVelocity,
        SpenglerGammeterBicycle,
    }

    controllers {
        MpcDistance,
        MpcPosition,
    }
}
