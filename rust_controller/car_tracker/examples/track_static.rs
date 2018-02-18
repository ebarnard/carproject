extern crate car_tracker;
extern crate cv;
extern crate image;

use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let frame = image::open("static_test_frame.tiff").unwrap().to_luma();

    let width = frame.width();
    let height = frame.height();

    let track_mask = image::open("static_test_track_mask.tiff")
        .unwrap()
        .to_luma()
        .into_raw();
    let bg = image::open("static_test_bg.tiff")
        .unwrap()
        .to_luma()
        .into_raw();
    let frame = frame.into_raw();

    let mut tracker = car_tracker::Tracker::new(width, height, &track_mask, &bg);

    // process the same frame a few times
    tracker.track_frame(&frame);
    tracker.track_frame(&frame);

    let start = Instant::now();
    let (x_median, y_median, theta) = tracker.track_frame(&frame);
    let dur = Instant::now() - start;
    println!(
        "took {}ms",
        dur.as_secs() as f64 * 1e3 + dur.subsec_nanos() as f64 * 1e-6
    );

    println!("med {} {} {}", x_median, y_median, theta * 180.0 / PI);

    let fg = cv::Mat::from_buffer(height as i32, width as i32, 0, &frame);
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

    fg.show("win", 0).unwrap();
}
