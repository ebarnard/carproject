// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate env_logger;
#[macro_use]
extern crate log;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate stats;
extern crate toml;

extern crate control_model;
extern crate controller_estimator;
extern crate prelude;
extern crate track;
extern crate visualisation;

mod config;
mod simulation_model;

use std::thread;
use std::time::Instant;

use controller_estimator::{Control, Measurement};
use prelude::*;
use visualisation::{Event, EventSender, Record};

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
    let (track, mut controllers) = controller_estimator::controllers_from_config();
    if controllers.len() != 1 {
        panic!("simulator can only be run with one car");
    }
    let mut controller = controllers.remove(0);

    let optimise_dt = controller.optimise_dt();
    let N = controller.N();

    let sim_config = config::SimulatorConfig::load();
    let R_sd: Vec<_> = sim_config.R.iter().cloned().map(float::sqrt).collect();
    let Q_state_sd: Vec<_> = sim_config
        .Q_state
        .iter()
        .cloned()
        .map(float::sqrt)
        .collect();

    let mut sim_model = simulation_model::model_from_config(&sim_config);

    record_tx
        .send(Event::Reset {
            n_history: 250,
            track: track.clone(),
            cars: vec![(controller.N(), controller.np())],
        })
        .expect("visualisation window closed");

    let n_steps = (sim_config.t / optimise_dt) as usize;
    let dt_duration = secs_to_duration(optimise_dt);
    let mut stats = stats::OnlineStats::new();
    let mut solve_time = Vec::with_capacity(n_steps);

    let mut s_prev = 0.0;
    let mut lap_start = 0;
    let mut lap_time_stats = stats::OnlineStats::new();

    let initial_position = track.nearest_centreline_point(0.0);
    let mut sim_state = sim_model.init_state(
        initial_position.x,
        initial_position.y,
        float::atan2(initial_position.dy_ds, initial_position.dx_ds),
    );

    let mut control = Control::default();

    for i in 0..n_steps {
        // Record control application
        // TODO: Move this back to bottom of loop once NLL becomes stable
        let control_time = dt_duration * i as u32;
        controller.control_applied(control, control_time);

        let step_start = Instant::now();

        // Run simulation
        sim_state = sim_model.step(optimise_dt, sim_state, &control);

        // Add noise to state
        for (x, &sd) in sim_state.0.iter_mut().zip(&Q_state_sd) {
            *x += randn() * sd;
        }

        // Add noise to measurement
        let state = sim_model.inspect_state(&sim_state);

        let s = track
            .centreline_distance(state.position.0, state.position.1)
            .unwrap();
        if s < s_prev {
            let lap_time = (i - lap_start) as float * optimise_dt * 1e3;
            if lap_time > 200.0 {
                println!("lap in {} ms", lap_time);
                if lap_start != 0 {
                    //break;
                }
                lap_start = i;
                lap_time_stats.add(lap_time);
            }
        }
        s_prev = s;

        let mut measurement = Measurement {
            position: (
                state.position.0 + randn() * R_sd[0],
                state.position.1 + randn() * R_sd[1],
            ),
            heading: state.heading + randn() * R_sd[2],
        };
        let measurement_time = dt_duration * i as u32;
        controller.measurement(Some(measurement), measurement_time);

        // Start controller timer
        let controller_start = Instant::now();

        // Run controller
        let control_application_time = dt_duration * (i + 1) as u32;
        let res = controller.step(control_application_time);

        // Stop timer
        let controller_millis = duration_to_secs(controller_start.elapsed()) * 1e3;
        stats.add(controller_millis);
        solve_time.push(controller_millis);

        control = res.control_horizon[0].0;

        info!("Controller took {} ms", controller_millis);
        info!("State {:?}", res.current_state);
        info!("Control {:?}", control);

        let position_horizon = res
            .control_horizon
            .iter()
            .map(|&(_, state)| state.position)
            .collect();

        record_tx
            .send(Event::Record(
                0,
                Record {
                    t: i as float * optimise_dt,
                    predicted_state: res.current_state,
                    control,
                    params: res.params.to_vec(),
                    param_var: res.param_var.to_vec(),
                    predicted_horizon: position_horizon,
                },
            ))
            .expect("visualisation window closed");

        let step_elapsed = Instant::now() - step_start;
        if let Some(step_remaining) = dt_duration.checked_sub(step_elapsed) {
            if sim_config.real_time {
                thread::sleep(step_remaining);
            }
        } else {
            println!(
                "step missed deadline. took {:.1}ms.",
                duration_to_secs(step_elapsed) * 1e3
            );
        }
    }

    solve_time.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "Running stats (mean/ms +/- stdev/ms, 99%/ms): {:?}, {}",
        stats,
        solve_time[(n_steps as float * 0.99) as usize]
    );

    println!("Lap times (mean/ms, stdev/ms): {:?}", lap_time_stats);
}

fn randn() -> float {
    use rand::Rng;
    rand::thread_rng().sample(rand::distributions::StandardNormal)
}
