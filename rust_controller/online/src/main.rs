// Ignore this lint otherwise many warnings are generated for common mathematical notation
#![allow(non_snake_case)]

extern crate env_logger;
#[macro_use]
extern crate log;
extern crate stats;

extern crate car_remote;
extern crate car_tracker;
extern crate controller_estimator;
extern crate prelude;
extern crate track;
extern crate track_aruco;

use nalgebra::{Matrix3, Vector3};
use std::io;
use std::time::{Duration, Instant};
use std::thread;

use prelude::*;
use controller_estimator::Measurement;
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
    let optimise_dt = controller.optimise_dt();
    let N = controller.N();

    record_tx
        .send(Event::Reset {
            n_history: 250,
            track: controller.track().clone(),
            horizon_len: N as usize,
            np: controller.np() as usize,
        })
        .expect("visualisation window closed");

    let dt_duration = Duration::new(
        optimise_dt.floor() as u64,
        (optimise_dt.fract() * 1e9) as u32,
    );
    let mut stats = stats::OnlineStats::new();

    info!("starting camera");
    let mut camera = car_tracker::Camera::new();
    let mut capture = camera.start_capture();
    let w = capture.width();
    let h = capture.height();
    info!("camera started");

    let H = track_aruco::find_homography(capture.latest_frame().1, w, h)
        .expect("cannot find world to image homography");
    debug!("H: {}", H);

    let H_inv = H.try_inverse().expect("H must be invertible");
    let H_inv = H_inv / H_inv[(2, 2)];

    let track_mask = controller.track().bitmap_mask(&H, w, h);
    let mut tracker = car_tracker::Tracker::new(w, h, &track_mask, capture.latest_frame().1);

    let mut remote = car_remote::Connection::new();
    remote.off(0);
    remote.on(0);

    println!("place car on track and press enter.");
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("could not read stdin");

    let start_time = Instant::now();

    for i in 0.. {
        // Start step timer
        let step_start = Instant::now();

        // Get the latest frame from the camera
        let (frame_time, frame) = capture.latest_frame();
        let frame_time = if frame_time > start_time {
            frame_time - start_time
        } else {
            Duration::from_secs(0)
        };

        // Get measurement of car's position in image coordinates
        let (image_x, image_y, image_heading) = tracker.track_frame(frame);
        info!("car at pixel {} {}", image_x, image_y);

        // Transform to world coordinates
        let measurement = image_to_world(&H_inv, image_x, image_y, image_heading);
        info!("car at {:?}", measurement);

        // Start controller timer
        let controller_start = Instant::now();

        // Run controller
        // TODO: Output control at regular intervals and skip optimising if a deadline is missed.
        let control_time = step_start - start_time + dt_duration;
        let res = controller.step(Some(measurement), frame_time, control_time);

        // Stop timer
        let dur = Instant::now().duration_since(controller_start);
        let millis = (dur.as_secs() as f64) * 1000.0 + f64::from(dur.subsec_nanos()) * 0.000_001;
        stats.add(millis);

        let control = res.horizon[0].0;

        info!("Controller took {} ms", millis);
        info!("State {:?}", res.predicted_state);
        info!("Control {:?}", control);

        let position_horizon = res.horizon
            .iter()
            .map(|&(_, state)| state.position)
            .collect();

        record_tx
            .send(Event::Record(Record {
                t: i as float * optimise_dt,
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

        // Send control to car
        remote.set(
            0,
            (control.throttle_position * 127.0) as i8,
            (control.steering_angle * 127.0) as i8,
        );
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
