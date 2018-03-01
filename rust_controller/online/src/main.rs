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
use controller_estimator::{Control, Measurement};
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
            n_cars: 1,
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
    let mut tracker = car_tracker::Tracker::new(1, w, h, &track_mask, capture.latest_frame().1);

    let mut remote = car_remote::Connection::new();
    remote.off(0);
    remote.on(0);

    println!("place car on track and press enter.");
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("could not read stdin");

    let mut control = Control::default();
    let mut prev_control_time = Duration::from_secs(0);
    let mut prev_frame_time = Duration::from_secs(0);

    // Synchronise the controller clock to the camera clock.
    let start_time = capture.latest_frame().0;

    for i in 0.. {
        // Send control input to the car.
        // TODO: Move this back to bottom of loop once NLL becomes stable.
        // `control_time` is recorded so it can be sent to the estimator.
        let control_time = start_time.elapsed();
        remote.set(
            0,
            (control.throttle_position * 127.0) as i8,
            (control.steering_angle * 127.0) as i8,
        );

        // Get the latest frame from the camera and Get measurement of car's position in image
        // coordinates.
        let (image_pos, frame_time) = loop {
            let (frame_time, frame) = capture.latest_frame();
            let frame_time = frame_time.duration_since(start_time);
            if frame_time > prev_frame_time && frame_time > prev_control_time {
                break (tracker.track_frame(frame)[0], frame_time);
            } else {
                // TODO: Use something better than a busy loop here.
                thread::yield_now();
            }
        };
        prev_frame_time = frame_time;
        prev_control_time = control_time;

        // Start the step timer once we've got a current frame.
        let step_start = start_time.elapsed();

        // Transform to world coordinates
        let mut measurement = image_pos.map(|p| {
            info!("car at pixel {} {}", p.x, p.y);

            let mut m = image_to_world(&H_inv, p.x, p.y, p.heading);

            // Assume car is pointing in the direction of the track.
            // TODO: Find the actual heading of the car.
            let s = controller
                .track()
                .centreline_distance(m.position.0, m.position.1);
            let cp = controller.track().nearest_centreline_point(s);
            if m.heading.cos() * cp.dx_ds + m.heading.sin() * cp.dy_ds < 0.0 {
                m.heading += ::std::f64::consts::PI;
            }

            info!("car at {:?}", m);
            m
        });

        // Ensure measurements and controls are recorded in temporal order.
        debug!("control time: {:?}", control_time);
        debug!("measurement time: {:?}", frame_time);
        if frame_time > control_time {
            controller.control_applied(control, control_time);
            controller.measurement(measurement, frame_time);
        } else {
            controller.measurement(measurement, frame_time);
            controller.control_applied(control, control_time);
        }

        // Start controller timer
        let controller_start = Instant::now();

        // Run controller
        // TODO: Output control at regular intervals and skip optimising if a deadline is missed.
        let control_application_time = step_start + dt_duration;
        debug!(
            "target control application time: {:?}",
            control_application_time
        );
        let res = controller.step(control_application_time);

        // Stop timer
        let controller_millis = duration_to_secs(controller_start.elapsed()) * 1e3;
        stats.add(controller_millis);

        control = res.control_horizon[0].0;

        info!("Controller took {} ms", controller_millis);
        info!("State {:?}", res.current_state);
        info!("Control {:?}", control);

        let position_horizon = res.control_horizon
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
                    // TODO: pass real variances here once the plots are working again
                    param_var: res.params.to_vec(),
                    predicted_horizon: position_horizon,
                },
            ))
            .expect("visualisation window closed");

        let step_elapsed = start_time.elapsed() - step_start;
        if let Some(step_remaining) = dt_duration.checked_sub(step_elapsed) {
            thread::sleep(step_remaining);
        } else {
            println!(
                "step missed deadline. took {:.1}ms.",
                duration_to_secs(step_elapsed) * 1e3
            );
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
