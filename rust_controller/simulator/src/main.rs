// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate env_logger;
#[macro_use]
extern crate log;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate stats;
extern crate toml;

extern crate control_model;
extern crate controller_estimator;
extern crate prelude;
extern crate track;

mod config;
mod simulation_model;

use std::time::{Duration, Instant};
use std::thread;

use prelude::*;
use controller_estimator::{Control, Measurement, State};
use controller_estimator::visualisation::{self, Event, EventSender, Record};

fn main() {
    env_logger::init();

    let (mut window, record_tx) = visualisation::new();
    let sim_handle = thread::spawn(move || run(record_tx));
    window.run_on_main_thread();

    if sim_handle.join().is_err() {
        error!("simulation failed");
    }

    flame_merge::write_flame();
}

fn run(mut record_tx: EventSender<Event>) {
    let mut controller = controller_estimator::controller_from_config();
    let dt = controller.dt();
    let N = controller.N();

    let sim_config = config::SimulatorConfig::load();

    let mut sim_model = simulation_model::model_from_config(&sim_config);

    record_tx
        .send(Event::Reset {
            n_history: 250,
            track: controller.track().clone(),
            horizon_len: N as usize,
            np: controller.np() as usize,
        })
        .expect("visualisation window closed");

    let n_steps = (sim_config.t / dt) as usize;
    let dt_duration = Duration::new(dt.floor() as u64, (dt.fract() * 1e9) as u32);
    let mut stats = stats::OnlineStats::new();

    let initial_position = controller.track().nearest_centreline_point(0.0);
    let mut state = State::default();
    state.position = (initial_position.x, initial_position.y);
    state.heading = float::atan2(initial_position.dy_ds, initial_position.dx_ds);

    let mut control = Control::default();

    for i in 0..n_steps {
        let step_start = Instant::now();

        // Run simulation
        state = sim_model.step(dt, &state, &control);

        // Add noise to measurement
        let mut measurement = Measurement {
            position: (
                state.position.0 + randn() * 0.002,
                state.position.1 + randn() * 0.002,
            ),
            heading: state.heading + randn() * 0.03,
        };

        // Start controller timer
        let controller_start = Instant::now();

        // Run controller
        let res = controller.step(Some(measurement));

        // Stop timer
        let dur = Instant::now().duration_since(controller_start);
        let millis = (dur.as_secs() as f64) * 1000.0 + f64::from(dur.subsec_nanos()) * 0.000_001;
        stats.add(millis);

        control = res.horizon[0].0;

        info!("Controller took {} ms", millis);
        info!("State {:?}", res.predicted_state);
        info!("Control {:?}", control);

        let position_horizon = res.horizon
            .iter()
            .map(|&(_, state)| state.position)
            .collect();

        record_tx
            .send(Event::Record(Record {
                t: i as float * dt,
                predicted_state: res.predicted_state,
                control,
                params: res.params.to_vec(),
                // TODO: pass real variances here once the plots are working again
                param_var: res.params.to_vec(),
                predicted_horizon: position_horizon,
            }))
            .expect("visualisation window closed");

        let step_elapsed = Instant::now() - step_start;
        if let Some(step_remaining) = dt_duration.checked_sub(step_elapsed) {
            thread::sleep(step_remaining);
        } else {
            let millis =
                step_elapsed.as_secs() as f64 * 1e3 + step_elapsed.subsec_nanos() as f64 / 1e6;
            println!("step missed deadline. took {:.1}ms.", millis);
        }
    }

    println!("Running stats (mean/ms, stdev/ms): {:?}", stats);
}
