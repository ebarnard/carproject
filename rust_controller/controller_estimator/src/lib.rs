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

use nalgebra::{self, DimAdd, DimSum, U0, U3, Vector3};
use std::sync::Arc;

use prelude::*;
use controller::Controller;
use control_model::{Control, ControlModel, State};
use estimator::{Estimator, JointEKF};
use track::Track;

pub use estimator::Measurement;

pub trait ControllerEstimator {
    fn dt(&self) -> float;
    fn N(&self) -> u32;
    fn np(&self) -> u32;
    fn track(&self) -> &Arc<Track>;

    fn step(&mut self, measurement: Option<Measurement>) -> StepResult;
}

pub struct StepResult<'a> {
    pub predicted_state: State,
    pub horizon: &'a [(Control, State)],
    pub params: &'a [float],
}

pub struct ControllerEstimatorImpl<M: 'static + ControlModel>
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
    controller: Box<Controller<M>>,
    estimator: Box<Estimator<M>>,
    params: Vector<M::NP>,
    prev_control: Vector<M::NI>,
    horizon: Vec<(Control, State)>,
}

pub fn new<M: 'static + ControlModel>() -> Box<ControllerEstimator>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<U3, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    let config = config::ControllerConfig::load();
    let track = Arc::new(track::Track::load(&*config.track));
    let model = M::new();
    let N = config.N;

    let mut controller = controller::MpcTime::<M>::new(&model, N, track.clone());
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
        controller: Box::new(controller) as Box<Controller<M>>,
        estimator: Box::new(state_estimator) as Box<Estimator<M>>,
        params: initial_params,
        prev_control: Vector::<M::NI>::zeros(),
        horizon: vec![Default::default(); N as usize],
    })
}

impl<M: 'static + ControlModel> ControllerEstimator for ControllerEstimatorImpl<M>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<U3, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    fn dt(&self) -> float {
        self.config.dt
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

    fn step(&mut self, measurement: Option<Measurement>) -> StepResult {
        // Estimate state and params
        let (predicted_state, p) = self.estimator.step(
            &mut self.model,
            self.config.dt,
            &self.prev_control,
            measurement,
            &self.params,
        );
        self.params = p;

        // Advance
        let next_predicted_state = self.model.step(
            self.config.dt,
            &predicted_state,
            &self.prev_control,
            &self.params,
        );

        // Run controller
        let (horizon_ctrl, horizon_state) = self.controller.step(
            &mut self.model,
            self.config.dt,
            &next_predicted_state,
            &self.params,
        );
        self.prev_control = horizon_ctrl.column(0).into_owned();

        for i in 0..(self.config.N as usize) {
            let control = self.model
                .u_to_control(&horizon_ctrl.column(i).into_owned());
            let state = self.model.x_to_state(&horizon_state.column(i).into_owned());
            self.horizon[i] = (control, state);
        }

        StepResult {
            predicted_state: self.model.x_to_state(&predicted_state),
            horizon: &self.horizon,
            params: &self.params.as_slice(),
        }
    }
}
