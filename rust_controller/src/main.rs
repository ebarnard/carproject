// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate csv;
extern crate flame;
extern crate nalgebra;
extern crate cubic_spline;
extern crate env_logger;
extern crate gnuplot;
extern crate itertools;
extern crate kdtree;
#[macro_use]
extern crate log;
extern crate sparse;
extern crate stats;

mod prelude;
mod control_model;
mod controller;
mod odeint;
mod simulation_model;
mod track;
mod visualisation;
mod osqp;

use std::time::Instant;
use std::panic::{self, AssertUnwindSafe};

use prelude::*;
use controller::Controller;
use simulation_model::{SimulationModel, State};
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

fn run(track: &track::Track, history: &mut History) {
    let model = control_model::SpenglerGammeterBicycle;
    let mut controller = controller::MpcPosition::new(50, model, &track);

    let mut sim_model = control_model::SpenglerGammeterBicycle;

    let mut state = State::default();
    state.position = (track.x[0], track.y[0]);

    run_simulation(20.0, 0.01, state, &mut sim_model, &mut controller, history);
}

fn run_simulation(
    t: float,
    dt: float,
    mut state: State,
    sim_model: &mut SimulationModel,
    controller: &mut Controller,
    history: &mut History,
) {
    let n_steps = (t / dt) as usize;
    let mut stats = stats::OnlineStats::new();

    for i in 0..n_steps {
        let start = Instant::now();
        let (control, _) =
            controller.step(dt, &state.to_controller_state(), sim_model.params());
        let dur = Instant::now().duration_since(start);
        let millis = (dur.as_secs() as f64) * 1000.0 + (dur.subsec_nanos() as f64) * 0.000_001;
        stats.add(millis);

        info!("Controller took {} ms", millis);
        info!("State {:?}", state);
        info!("Control {:?}", control);

        history.record(i as float * dt, &state);
        let next_state = sim_model.step(dt, &state, &control);

        state = next_state;
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
