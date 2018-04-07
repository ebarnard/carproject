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
extern crate visualisation;

use nalgebra::{Matrix3, Vector3};
use std::io;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};
use std::thread;

use prelude::*;
use controller_estimator::{Control, ControllerEstimator, Measurement};
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
    let (track, controllers) = controller_estimator::controllers_from_config();
    let n_cars = controllers.len() as u32;

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

    // Generate a bitmap mask of the track in image coordinates.
    let track_mask = track.track.bitmap_mask(&H, w, h);

    let mut tracker =
        car_tracker::Tracker::new(n_cars, w, h, &track_mask, capture.latest_frame().1);

    // Setup the remote controller.
    let remote = init_remote(n_cars);

    // Start the vis
    record_tx
        .send(Event::Reset {
            n_history: 250,
            track: track.clone(),
            horizon_len: 45, //controller.N() as usize,
            np: 7,           //controller.np() as usize,
            n_cars,
        })
        .expect("visualisation window closed");

    // Wait for cars to be placed on track.
    println!("place cars on track. cars must be placed left to right in car number order. press enter to continue.");
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("could not read stdin");

    // Synchronise the controller clock to the camera clock.
    let start_time = capture.latest_frame().0;
    let mut measurements = vec![None; n_cars as usize];

    // Create measurement triple buffers.
    track_latest_frame(
        &mut capture,
        start_time,
        &mut tracker,
        &H_inv,
        &track,
        &mut measurements,
    );

    // Reorder measurement_outputs so they correspond to car numbers.
    // Assume that cars start positioned left to right.
    let mut measurement_indices: Vec<_> = measurements
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m.expect("all cars must be detected before starting")))
        .collect();
    measurement_indices.sort_unstable_by(|a, b| {
        let a_x = a.1.position.0;
        let b_x = b.1.position.0;
        a_x.partial_cmp(&b_x).expect("unexpected nan")
    });

    // Start controller threads.
    let (mut measurement_txs, controller_thread_handles): (Vec<_>, Vec<_>) = controllers
        .into_iter()
        .enumerate()
        .map(|(i, controller)| {
            let (measurement_tx, measurement_rx) = channel();
            let remote = remote.clone();
            let record_tx = record_tx.clone();
            let thread_handle = thread::spawn(move || {
                controller_loop(
                    measurement_rx,
                    controller,
                    remote,
                    i as u32,
                    start_time,
                    record_tx,
                );
            });
            (measurement_tx, thread_handle)
        })
        .unzip();

    // Process camera frames on this thread.
    loop {
        let frame_time = track_latest_frame(
            &mut capture,
            start_time,
            &mut tracker,
            &H_inv,
            &track,
            &mut measurements,
        );
        let mut ordered_measurements = vec![None; measurements.len()];
        for (&measurement, &(i, _)) in measurements.iter().zip(&measurement_indices) {
            ordered_measurements[i] = measurement;
        }
        let ordered_measurements = Arc::new(ordered_measurements);

        measurement_txs.retain(|tx| tx.send((ordered_measurements.clone(), frame_time)).is_ok());
        if measurement_txs.len() == 0 {
            println!("all controller threads exited.");
            break;
        }
    }

    // Ensure the controller threads have exited.
    for handle in controller_thread_handles {
        let _ = handle.join();
    }
}

fn track_latest_frame(
    capture: &mut car_tracker::Capture,
    start_time: Instant,
    tracker: &mut car_tracker::Tracker,
    H_inv: &Matrix3<float>,
    track: &track::TrackAndLookup,
    measurements: &mut [Option<Measurement>],
) -> Duration {
    // Get the latest frame from the camera.
    let (frame_time, frame) = capture.latest_frame();
    let frame_time = frame_time.duration_since(start_time);

    // Find the car's position in image coordinates.
    let car_image_positions = tracker.track_frame(frame);

    assert_eq!(car_image_positions.len(), measurements.len());

    for (maybe_p, measurement) in car_image_positions.iter().zip(measurements) {
        *measurement = maybe_p.and_then(|p| {
            info!("car at pixel {} {}", p.x, p.y);
            // Transform car position from image to world coordinates.
            let mut m = image_to_world(&H_inv, p.x, p.y, p.heading);

            if let Some(s) = track.centreline_distance(m.position.0, m.position.1) {
                let cp = track.nearest_centreline_point(s);

                // Assume car is pointing in the direction of the track.
                // TODO: Find the actual heading of the car.
                if m.heading.cos() * cp.dx_ds + m.heading.sin() * cp.dy_ds < 0.0 {
                    m.heading += ::std::f64::consts::PI;
                }

                info!("car at {:?}", m);
                Some(m)
            } else {
                // Exclude any measurements outside the track bounds.
                None
            }
        });
    }
    frame_time
}

fn controller_loop(
    measurement_rx: Receiver<(Arc<Vec<Option<Measurement>>>, Duration)>,
    mut controller: Box<ControllerEstimator>,
    remote: Arc<Mutex<car_remote::Connection>>,
    car_index: u32,
    start_time: Instant,
    mut record_tx: EventSender<Event>,
) {
    let optimise_dt = controller.optimise_dt();
    let dt_duration = Duration::new(
        optimise_dt.floor() as u64,
        (optimise_dt.fract() * 1e9) as u32,
    );

    let mut stats = stats::OnlineStats::new();

    let mut control = Control::default();
    let mut prev_control_time = Duration::from_secs(0);
    let mut prev_measurement_time = Duration::from_secs(0);

    for i in 0.. {
        // Send control input to the car.
        // TODO: Move this back to bottom of loop once NLL becomes stable.
        // `control_time` is recorded so it can be sent to the estimator.
        let control_time = start_time.elapsed();
        remote.lock().expect("controller poisoned").set(
            car_index as u8,
            (control.throttle_position * 127.0) as i8,
            (control.steering_angle * 127.0) as i8,
        );

        // Get the latest frame from the camera and Get measurement of car's position in image
        // coordinates.
        let (measurement, measurement_time) = loop {
            let (measurements, measurement_time) =
                measurement_rx.recv().expect("measurement thread exited");
            if measurement_time > prev_measurement_time && measurement_time > prev_control_time {
                break (measurements[car_index as usize], measurement_time);
            }
        };
        prev_measurement_time = measurement_time;
        prev_control_time = control_time;

        // Start the step timer once we've got a current frame.
        let step_start = start_time.elapsed();

        // Ensure measurements and controls are recorded in temporal order.
        debug!("control time: {:?}", control_time);
        debug!("measurement time: {:?}", measurement_time);
        if measurement_time > control_time {
            controller.control_applied(control, control_time);
            controller.measurement(measurement, measurement_time);
        } else {
            controller.measurement(measurement, measurement_time);
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
        println!("params {:?}", res.params);

        let position_horizon = res.control_horizon
            .iter()
            .map(|&(_, state)| state.position)
            .collect();

        record_tx
            .send(Event::Record(
                car_index,
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

fn init_remote(n_cars: u32) -> Arc<Mutex<car_remote::Connection>> {
    let mut remote = car_remote::Connection::new();
    // Turn on the cars.
    for i in 0..n_cars {
        remote.set(i as u8, 0, 0);
        remote.on(i as u8);
    }
    Arc::new(Mutex::new(remote))
}
