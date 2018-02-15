// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate config;
extern crate control_model;
extern crate controller;
extern crate env_logger;
extern crate estimator;
#[macro_use]
extern crate log;
extern crate prelude;
extern crate stats;
extern crate track;
extern crate ui;

mod flame_merge;
mod simulation_model;
mod visualisation;

use nalgebra::{Vector3, Vector4, Vector6};
use std::time::{Duration, Instant};
use std::thread;
use std::sync::Arc;

use prelude::*;
use controller::Controller;
use control_model::ControlModel;
use simulation_model::{SimulationModel, State};
use track::Track;
use estimator::{Estimator, JointEKF, Measurement};
use ui::EventSender;
use visualisation::{Event, Record};

fn main() {
    env_logger::init();

    let config = config::load();
    let track = Arc::new(track::Track::load(&*config.track));

    let (mut window, record_tx) = visualisation::new(250, track.clone());
    let sim_handle = thread::spawn(move || run(&config, track, record_tx));
    window.run_on_main_thread();

    if sim_handle.join().is_err() {
        error!("simulation failed");
    }

    flame_merge::write_flame();
}

type Model = control_model::SpenglerGammeterBicycle;

fn run(config: &config::Config, track: Arc<Track>, mut record_tx: EventSender<Event>) {
    let model = control_model::SpenglerGammeterBicycle;

    let mut controller =
        controller::MpcTime::<Model>::new(&model, config.controller.N, track.clone());

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

    record_tx
        .send(Event::Reset {
            horizon_len: config.controller.N as usize,
            np: Q_params.shape().0,
        })
        .expect("visualisation window closed");

    run_simulation(
        config.t,
        config.dt,
        config.simulator.real_time,
        state,
        &mut *sim_model,
        &model,
        &mut controller,
        initial_params,
        &mut state_estimator,
        &mut record_tx,
    );
}

fn run_simulation<M: ControlModel>(
    t: float,
    dt: float,
    real_time: bool,
    mut prev_state: State,
    sim_model: &mut SimulationModel,
    model: &M,
    controller: &mut Controller<M>,
    mut params: Vector<M::NP>,
    estimator: &mut Estimator<M>,
    record_tx: &mut EventSender<Event>,
) where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    let n_steps = (t / dt) as usize;
    let dt_duration = Duration::new(dt.floor() as u64, (dt.fract() * 1e9) as u32);
    let mut stats = stats::OnlineStats::new();

    let mut control: Vector<M::NI> = nalgebra::zero();

    for i in 0..n_steps {
        let step_start = Instant::now();

        // Run simulation
        let ctrl = model.u_to_control(&control);
        let state = sim_model.step(dt, &prev_state, &ctrl);

        // Add noise to measurement
        let mut measurement = Measurement {
            position: (
                state.position.0 + randn() * 0.002,
                state.position.1 + randn() * 0.002,
            ),
            heading: state.heading + randn() * 0.03,
        };

        // Start timer
        let start = Instant::now();

        // Estimate state and params
        let (predicted_state, p) = estimator.step(model, dt, &control, Some(measurement), &params);
        params = p;

        // Run controller
        let (horizon_ctrl, horizon_state) = controller.step(model, dt, &predicted_state, &params);
        control = horizon_ctrl.column(0).into_owned();

        // Stop timer
        let dur = Instant::now().duration_since(start);
        let millis = (dur.as_secs() as f64) * 1000.0 + f64::from(dur.subsec_nanos()) * 0.000_001;
        stats.add(millis);

        info!("Controller took {} ms", millis);
        info!("State {:?}", state);
        info!("Control {:?}", model.u_to_control(&control));

        let mut horizon = Vec::with_capacity(horizon_state.shape().1);
        for i in 0..horizon.capacity() {
            let col = horizon_state.column(i);
            horizon.push((col[0], col[1]));
        }

        record_tx
            .send(Event::Record(Record {
                t: i as float * dt,
                state: state.clone(),
                predicted_state: model.x_to_state(&predicted_state),
                control: model.u_to_control(&control),
                params: params.as_slice().to_vec(),
                param_var: estimator.param_covariance().diagonal().as_slice().to_vec(),
                predicted_horizon: horizon,
            }))
            .expect("visualisation window closed");

        prev_state = state;

        let step_elapsed = Instant::now() - step_start;
        if let Some(step_remaining) = dt_duration.checked_sub(step_elapsed) {
            if real_time {
                thread::sleep(step_remaining);
            }
        } else {
            let millis =
                step_elapsed.as_secs() as f64 * 1e3 + step_elapsed.subsec_nanos() as f64 / 1e6;
            println!("step missed deadline. took {:.1}ms.", millis);
        }
    }

    println!("Running stats (mean/ms, stdev/ms): {:?}", stats);
}
