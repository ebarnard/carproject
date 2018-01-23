// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate config;
extern crate csv;
extern crate cubic_spline;
extern crate env_logger;
extern crate flame;
extern crate gnuplot;
extern crate itertools;
extern crate kdtree;
#[macro_use]
extern crate log;
extern crate nalgebra;
extern crate rand;
extern crate sparse;
extern crate stats;

mod prelude;
mod control_model;
mod controller;
mod estimator;
mod flame_merge;
mod odeint;
mod simulation_model;
mod track;
mod visualisation;
mod osqp;

use nalgebra::{Vector3, Vector4, Vector6};
use std::time::Instant;
use std::panic::{self, AssertUnwindSafe};

use prelude::*;
use controller::Controller;
use control_model::ControlModel;
use simulation_model::{SimulationModel, State};
use estimator::{Estimator, JointEKF, Measurement};
use visualisation::History;

fn main() {
    env_logger::init().expect("logger init failed");

    let config = config::load();
    let track = track::Track::load(&*config.track);
    let n_steps = (config.t / config.dt) as usize;
    let mut history = History::new(n_steps);

    if panic::catch_unwind(AssertUnwindSafe(|| run(&config, &track, &mut history))).is_err() {
        error!("simulation failed");
    }

    visualisation::plot(&track, &history);

    flame_merge::write_flame();
}

type Model = control_model::SpenglerGammeterBicycle;

fn run(config: &config::Config, track: &track::Track, history: &mut History) {
    let model = control_model::SpenglerGammeterBicycle;

    let mut controller = controller::MpcTime::<Model>::new(&model, 50, track);

    let initial_params = Vector6::from_column_slice(&config.controller.initial_params);
    let Q_state = Matrix::from_diagonal(&Vector4::from_column_slice(&config.controller.Q_state));
    let Q_initial_params = Matrix::from_diagonal(&Vector6::from_column_slice(
        &config.controller.Q_initial_params,
    ));
    let Q_params = Q_initial_params * config.controller.Q_params_multiplier;
    let R = Matrix::from_diagonal(&Vector3::from_column_slice(&config.controller.R));

    let mut state_estimator = JointEKF::<Model>::new(Q_state, Q_params, Q_initial_params, R);

    let initial_position = track.nearest_centreline_point(0.0);
    let mut state = State::default();
    state.position = (initial_position.x, initial_position.y);

    let mut sim_model = simulation_model::model_from_config(&config.simulator);

    run_simulation(
        config.t,
        config.dt,
        state,
        &mut *sim_model,
        &model,
        &mut controller,
        initial_params,
        &mut state_estimator,
        history,
    );
}

fn run_simulation<M: ControlModel>(
    t: float,
    dt: float,
    mut prev_state: State,
    sim_model: &mut SimulationModel,
    model: &M,
    controller: &mut Controller<M>,
    mut params: Vector<M::NP>,
    estimator: &mut Estimator<M>,
    history: &mut History,
) where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    let n_steps = (t / dt) as usize;
    let mut stats = stats::OnlineStats::new();

    let mut control: Vector<M::NI> = nalgebra::zero();

    for i in 0..n_steps {
        // Run simulation
        let ctrl = model.u_to_control(&control);
        let state = sim_model.step(dt, &prev_state, &ctrl);
        let mut measurement = Measurement::from_state(&state);

        // Add noise to measurement
        measurement.position.0 += randn() * 0.002;
        measurement.position.1 += randn() * 0.002;
        measurement.heading += randn() * 0.03;

        // Start timer
        let start = Instant::now();

        // Estimate state and params
        let (predicted_state, p) = estimator.step(model, dt, &control, Some(measurement), &params);
        params = p;

        // Run controller
        let (ctrl, _) = controller.step(model, dt, &predicted_state, &params);
        control = ctrl;

        // Stop timer
        let dur = Instant::now().duration_since(start);
        let millis = (dur.as_secs() as f64) * 1000.0 + f64::from(dur.subsec_nanos()) * 0.000_001;
        stats.add(millis);

        info!("Controller took {} ms", millis);
        info!("State {:?}", state);
        info!("Control {:?}", model.u_to_control(&control));

        history.record(
            i as float * dt,
            &state,
            &model.x_to_state(&predicted_state),
            model.u_to_control(&control),
            &params,
            &estimator.param_covariance(),
        );

        prev_state = state;
    }

    println!("Running stats (mean/ms, stdev/ms): {:?}", stats);
}
