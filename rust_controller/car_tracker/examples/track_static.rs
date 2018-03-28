extern crate car_tracker;
extern crate cv;
extern crate image;

mod utils;

use cv::highgui::Show;
use std::time::Instant;

fn main() {
    let frame = image::open("static_test_frame_one_car.tiff")
        .unwrap()
        .to_luma();

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

    let mut tracker = car_tracker::Tracker::new(1, width, height, &track_mask, &bg);

    let start = Instant::now();
    let car_positions = tracker.track_frame(&frame);
    let dur = Instant::now() - start;

    println!(
        "took {}ms",
        dur.as_secs() as f64 * 1e3 + dur.subsec_nanos() as f64 * 1e-6
    );

    let mut fg = cv::Mat::from_buffer(height as i32, width as i32, 0, &frame);
    utils::draw_car_positions(car_positions, &mut fg);
    fg.show("win", 0).unwrap();
}
