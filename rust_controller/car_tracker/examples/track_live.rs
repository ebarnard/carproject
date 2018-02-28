extern crate car_tracker;
extern crate cv;
extern crate image;

mod utils;

use std::env;
use std::time::Instant;

fn main() {
    let max_cars = parse_cars_from_command_line();

    let mut camera = car_tracker::Camera::new();
    let mut cap = camera.start_capture();
    let frame_bytes = cap.bytes();
    let frame_width = cap.width();
    let frame_height = cap.height();
    let mut frame = vec![0; frame_bytes];

    let track_mask = vec![1; frame_bytes];

    let mut bg = vec![0; frame_bytes];
    bg.copy_from_slice(cap.latest_frame().1);

    let mut tracker =
        car_tracker::Tracker::new(max_cars, frame_width, frame_height, &track_mask, &bg);

    for i in 0.. {
        frame.copy_from_slice(cap.latest_frame().1);

        let start = Instant::now();
        let car_positions = tracker.track_frame(&frame);
        let dur = Instant::now() - start;

        // Only print and show sometimes
        if i % 20 != 0 {
            continue;
        }

        println!(
            "took {}ms",
            dur.as_secs() as f64 * 1e3 + dur.subsec_nanos() as f64 * 1e-6
        );

        let mut fg = cv::Mat::from_buffer(frame_height as i32, frame_width as i32, 0, &frame);
        utils::draw_car_positions(car_positions, &mut fg);
        fg.show("win", 1).unwrap();
    }
}

pub fn parse_cars_from_command_line() -> u32 {
    let mut args = env::args();
    // Skip executable path.
    args.next();
    if let Some("--cars") = args.next().as_ref().map(|a| a.as_str()) {
        if let Some(cars) = args.next().and_then(|c| c.parse().ok()) {
            return cars;
        }
    }
    panic!("error. maximum number of cars must be specified by passing --cars <N>.");
}
