// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

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
mod flame_merge;
mod param_estimator;
mod odeint;
mod simulation_model;
mod state_estimator;
mod track;
mod visualisation;
mod osqp;

use nalgebra::{Vector3, Vector4, Vector6};
use rand::distributions::normal::StandardNormal;
use std::time::Instant;
use std::panic::{self, AssertUnwindSafe};

use prelude::*;
use controller::Controller;
use control_model::{ControlModel};
use simulation_model::{SimulationModel, State};
use state_estimator::StateEstimator;
use visualisation::History;

fn main() {
    env_logger::init().expect("logger init failed");

    let track = track::Track::load("data/tracks/3yp_track2500.csv");
    let mut history = History::new(0);

    if panic::catch_unwind(AssertUnwindSafe(|| run(&track, &mut history))).is_err() {
        error!("simulation failed");
    }

    visualisation::plot(&track, &history);

    flame_merge::write_flame();
}

type Model = control_model::SpenglerGammeterBicycle;

fn run(track: &track::Track, history: &mut History) {
    let model = control_model::SpenglerGammeterBicycle;
    let mut controller = controller::MpcPosition::<Model>::new(&model, 50, &track);

    let params = Vector6::new(0.5, 2.0, 2.0, 1.5, 0.0, 0.0045);
    let delta_max = Vector6::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0) * 0.01;
    let initial_params = &params + &delta_max * 50.0;

    let Q = Matrix::from_diagonal(&Vector4::new(0.000005, 0.000005, 0.0005, 0.00001));
    let Q_params = Matrix::from_diagonal(&Vector6::new(0.05, 2.0, 0.5, 0.4, 0.0, 0.0));
    let R = Matrix::from_diagonal(&Vector3::new(0.000004, 0.000004, 0.0009));

    let mut state_estimator = state_estimator::StateAndParameterEKF::<Model>::new(Q, Q_params, R);

    let mut state = State::default();
    state.position = (track.x[0], track.y[0]);

    run_simulation(
        90.0,
        0.01,
        state,
        &mut simulation_model::SpenglerGammeterBicycle::new(params),
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
    state_estimator: &mut StateEstimator<M>,
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
        let mut measurement = state_estimator::Measurement::from_state(&state);

        // Add noise to measurement
        let StandardNormal(x_noise) = rand::random();
        let StandardNormal(y_noise) = rand::random();
        let StandardNormal(heading_noise) = rand::random();
        measurement.position.0 += x_noise * 0.002;
        measurement.position.1 += y_noise * 0.002;
        measurement.heading += heading_noise * 0.03;

        // Start timer
        let start = Instant::now();

        // Estimate state and params
        let (predicted_state, p) =
            state_estimator.step(model, dt, &control, Some(measurement), &params);
        params = p;

        // Run controller
        let (ctrl, _) = controller.step(model, dt, &predicted_state, &params);
        control = ctrl;

        // Stop timer
        let dur = Instant::now().duration_since(start);
        let millis = (dur.as_secs() as f64) * 1000.0 + (dur.subsec_nanos() as f64) * 0.000_001;
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
            &state_estimator.param_covariance(),
        );

        prev_state = state;
    }

    println!("Running stats (mean/ms, stdev/ms): {:?}", stats);
}
