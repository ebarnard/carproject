// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate car_remote;
extern crate car_tracker;
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
extern crate track_aruco;
extern crate ui;

mod visualisation;

use nalgebra::{Matrix3, Vector3, Vector4, Vector6};
use std::io;
use std::time::{Duration, Instant};
use std::thread;
use std::sync::Arc;

use prelude::*;
use controller::Controller;
use control_model::ControlModel;
use estimator::{Estimator, JointEKF, Measurement};
use track::Track;
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
    let u_min = Vector2::from_column_slice(&config.controller.u_min);
    let u_max = Vector2::from_column_slice(&config.controller.u_max);
    controller.update_input_bounds(u_min, u_max);

    let initial_params = Vector6::from_column_slice(&config.controller.initial_params);
    let Q_state = Matrix::from_diagonal(&Vector4::from_column_slice(&config.controller.Q_state));
    let Q_initial_params = Matrix::from_diagonal(&Vector6::from_column_slice(
        &config.controller.Q_initial_params,
    ));
    let Q_params = Q_initial_params * config.controller.Q_params_multiplier;
    let R = Matrix::from_diagonal(&Vector3::from_column_slice(&config.controller.R));

    let mut state_estimator = JointEKF::<Model>::new(Q_state, Q_params, Q_initial_params, R);

    record_tx
        .send(Event::Reset {
            horizon_len: config.controller.N as usize,
            np: Q_params.shape().0,
        })
        .expect("visualisation window closed");

    run_simulation(
        config.dt,
        &track,
        &model,
        &mut controller,
        initial_params,
        &mut state_estimator,
        &mut record_tx,
    );
}

fn run_simulation<M: ControlModel>(
    dt: float,
    track: &Track,
    model: &M,
    controller: &mut Controller<M>,
    mut params: Vector<M::NP>,
    estimator: &mut Estimator<M>,
    record_tx: &mut EventSender<Event>,
) where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    let dt_duration = Duration::new(dt.floor() as u64, (dt.fract() * 1e9) as u32);
    let mut stats = stats::OnlineStats::new();

    info!("starting camera");
    let mut camera = car_tracker::Camera::new();
    let mut capture = camera.start_capture();
    let w = capture.width();
    let h = capture.height();
    info!("camera started");

    let H = track_aruco::find_homography(capture.latest_frame(), w, h)
        .expect("cannot find world to image homography");
    debug!("H: {}", H);

    let H_inv = H.try_inverse().expect("H must be invertible");
    let H_inv = H_inv / H_inv[(2, 2)];

    let track_mask = track.bitmap_mask(&H, w, h);
    let mut tracker = car_tracker::Tracker::new(w, h, &track_mask, capture.latest_frame());

    let mut control: Vector<M::NI> = nalgebra::zero();

    let mut remote = car_remote::Connection::new();
    remote.off(0);
    remote.on(0);

    println!("place car on track and press enter.");
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("could not read stdin");

    for i in 0.. {
        // Start step timer
        let step_start = Instant::now();

        // Get measurement of car's position in image coordinates
        let frame = capture.latest_frame();
        let (image_x, image_y, image_heading) = tracker.track_frame(frame);
        println!("car at pixel {} {}", image_x, image_y);

        // Transform to world coordinates
        let measurement = image_to_world(&H_inv, image_x, image_y, image_heading);
        println!("car at {:?}", measurement);

        // Start controller timer
        let controller_start = Instant::now();

        // Estimate state and params
        let (predicted_state, p) = estimator.step(model, dt, &control, Some(measurement), &params);
        params = p;

        // Run controller
        let (horizon_ctrl, horizon_state) = controller.step(model, dt, &predicted_state, &params);
        control = horizon_ctrl.column(0).into_owned();

        // Stop timer
        let dur = Instant::now().duration_since(controller_start);
        let millis = (dur.as_secs() as f64) * 1000.0 + f64::from(dur.subsec_nanos()) * 0.000_001;
        stats.add(millis);

        // Send control to car
        remote.set(0, (control[0] * 127.0) as i8, (control[1] * 127.0) as i8);

        info!("Controller took {} ms", millis);
        info!("State {:?}", predicted_state);
        info!("Control {:?}", model.u_to_control(&control));

        let mut horizon = Vec::with_capacity(horizon_state.shape().1);
        for i in 0..horizon.capacity() {
            let col = horizon_state.column(i);
            horizon.push((col[0], col[1]));
        }

        record_tx
            .send(Event::Record(Record {
                t: i as float * dt,
                predicted_state: model.x_to_state(&predicted_state),
                control: model.u_to_control(&control),
                params: params.as_slice().to_vec(),
                param_var: estimator.param_covariance().diagonal().as_slice().to_vec(),
                predicted_horizon: horizon,
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

fn image_to_world(
    H_inv: &Matrix3<float>,
    image_x: float,
    image_y: float,
    image_heading: float,
) -> Measurement {
    let image_pos = Vector3::new(image_x, image_y, 1.0);
    let (sin_heading, cos_heading) = image_heading.sin_cos();
    let image_heading = Vector3::new(cos_heading, sin_heading, 0.0);

    let world_pos = H_inv * image_pos;
    let world_pos = world_pos / world_pos[2];

    let world_heading = H_inv * image_heading;
    let world_heading = float::atan2(world_heading[1], world_heading[0]);

    Measurement {
        position: (world_pos[0], world_pos[1]),
        heading: world_heading,
    }
}
