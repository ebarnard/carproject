extern crate car_tracker;
extern crate cv;
extern crate image;

use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let mut camera = car_tracker::Camera::new();
    let mut cap = camera.start_capture();
    let frame_bytes = cap.bytes();
    let frame_width = cap.width();
    let frame_height = cap.height();
    let mut frame = vec![0; frame_bytes];

    let track_mask = vec![1; frame_bytes];

    let mut bg = vec![0; frame_bytes];
    cap.wait_latest_frame(&mut bg[..]);
    cap.wait_latest_frame(&mut bg[..]);
    cap.wait_latest_frame(&mut bg[..]);

    let mut tracker = car_tracker::Tracker::new(frame_width, frame_height, &track_mask, &bg);

    for i in 0.. {
        cap.wait_latest_frame(&mut frame[..]);

        let start = Instant::now();
        let (x_median, y_median, theta) = tracker.track_frame(&frame);
        let dur = Instant::now() - start;

        // Only print and show sometimes
        if i % 20 != 0 {
            continue;
        }

        println!(
            "took {}ms",
            dur.as_secs() as f64 * 1e3 + dur.subsec_nanos() as f64 * 1e-6
        );
        println!("med {} {} {}", x_median, y_median, theta * 180.0 / PI);

        let fg = cv::Mat::from_buffer(frame_height as i32, frame_width as i32, 0, &frame);
        fg.rectangle_custom(
            cv::Rect::new(x_median as i32 - 5, y_median as i32 - 5, 10, 10),
            cv::Scalar::all(127),
            5,
            cv::LineTypes::Filled,
        );

        let x = (theta.cos() * 55.0 + x_median as f64) as i32;
        let y = (theta.sin() * 55.0 + y_median as f64) as i32;
        fg.line_custom(
            cv::Point2i::new(x_median as i32, y_median as i32),
            cv::Point2i::new(x, y),
            cv::Scalar::all(127),
            3,
            cv::LineTypes::Filled,
            0,
        );

        fg.show("win", 1).unwrap();
    }
}
