use car_tracker;
use cv;

pub fn draw_car_positions(positions: &[Option<car_tracker::Car>], frame: &mut cv::Mat) {
    for p in positions.iter().filter_map(Option::as_ref) {
        frame.rectangle_custom(
            cv::Rect::new(p.x as i32 - 5, p.y as i32 - 5, 10, 10),
            cv::Scalar::all(127),
            5,
            cv::LineTypes::Filled,
        );

        let x = (p.heading.cos() * 55.0 + p.x) as i32;
        let y = (p.heading.sin() * 55.0 + p.y) as i32;
        frame.line_custom(
            cv::Point2i::new(p.x as i32, p.y as i32),
            cv::Point2i::new(x, y),
            cv::Scalar::all(127),
            3,
            cv::LineTypes::Filled,
            0,
        );
    }
}
