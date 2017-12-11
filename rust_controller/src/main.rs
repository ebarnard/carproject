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
use control_model::{ControlModel, SimulationControlModel, SpenglerGammeterBicycle};
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

    write_flame();
}

type Model = SpenglerGammeterBicycle;

fn run(track: &track::Track, history: &mut History) {
    let mut controller = controller::MpcPosition::<Model>::new(50, &track);

    let delta_max = Vector6::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0) * 0.01;
    let initial_params = Vector6::from_column_slice(Model::default_params()) + &delta_max * 50.0;

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
        &mut SpenglerGammeterBicycle,
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

    let true_params = Vector::<M::NP>::from_column_slice(sim_model.params());

    for i in 0..n_steps {
        // Run simulation
        let state = sim_model.step(dt, &prev_state, &M::u_to_control(&control));
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
        let (predicted_state, p) = state_estimator.step(dt, &control, Some(measurement), &params);
        params = p;

        // Run controller
        let (ctrl, _) = controller.step(dt, &predicted_state, &params);
        control = ctrl;

        // Stop timer
        let dur = Instant::now().duration_since(start);
        let millis = (dur.as_secs() as f64) * 1000.0 + (dur.subsec_nanos() as f64) * 0.000_001;
        stats.add(millis);

        info!("Controller took {} ms", millis);
        info!("State {:?}", state);
        info!("Control {:?}", M::u_to_control(&control));

        let param_err = (&true_params - &params).norm();
        history.record(i as float * dt, &state, &M::x_to_state(&predicted_state), param_err);

        prev_state = state;
    }

    println!("Running stats (mean/ms, stdev/ms): {:?}", stats);
}

fn write_flame() {
    let mut spans = flame::threads().into_iter().next().unwrap().spans;
    merge_spans(&mut spans);

    use std::fs::File;
    flame::dump_html_custom(&mut File::create("flame-graph.html").unwrap(), &spans).unwrap();
}

use std::mem;
use std::usize;

fn merge_spans(spans: &mut Vec<flame::Span>) {
    if spans.is_empty() {
        return;
    }

    // Sort so spans to be merged are adjacent and spans with the most children are merged into
    spans.sort_unstable_by(|s1, s2| {
        let a = (&s1.name, s1.depth, usize::MAX - s1.children.len());
        let b = (&s2.name, s2.depth, usize::MAX - s2.children.len());
        a.cmp(&b)
    });

    // Copy children and sum delta from spans to be merged
    let mut merge_targets = vec![0];
    {
        let mut spans_iter = spans.iter_mut().enumerate();
        let (_, mut current) = spans_iter.next().unwrap();
        for (i, span) in spans_iter {
            if current.name == span.name && current.depth == span.depth {
                current.delta += span.delta;
                let mut children = mem::replace(&mut span.children, Vec::new());
                current.children.extend(children.into_iter());
            } else {
                current = span;
                merge_targets.push(i);
            }
        }
    }

    // Move merged spans to the front of the spans vector
    for (target_i, &current_i) in merge_targets.iter().enumerate() {
        spans.swap(target_i, current_i);
    }

    // Remove duplicate spans
    spans.truncate(merge_targets.len());

    // Merge children of the newly collapsed spans
    for span in spans {
        merge_spans(&mut span.children);
    }
}
