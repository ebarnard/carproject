extern crate car_tracker;
extern crate image;

#[test]
fn static_one_car() {
    let (width, height, frame, bg, track_mask) = open_images("static_test_frame_one_car.tiff");

    let mut tracker = car_tracker::Tracker::new(1, width, height, &track_mask, &bg);
    let car_positions = tracker.track_frame(&frame);

    assert_eq!(car_positions.len(), 1);

    if let Some(p) = car_positions[0] {
        assert_eq!(p.x, 1013.0);
        assert_eq!(p.y, 474.0);
        let expected_heading = -1.2698372461642782;
        assert!((p.heading - expected_heading).abs() < 1e-5);
    } else {
        panic!("car not found");
    }
}

#[test]
fn static_two_cars() {
    let (width, height, frame, bg, track_mask) = open_images("static_test_frame_two_cars.tiff");

    let mut tracker = car_tracker::Tracker::new(2, width, height, &track_mask, &bg);
    let car_positions = tracker.track_frame(&frame);

    assert_eq!(car_positions.len(), 2);

    if let Some(p) = car_positions[0] {
        assert_eq!(p.x, 1013.0);
        assert_eq!(p.y, 474.0);
        let expected_heading = -1.2698372461642782;
        assert!((p.heading - expected_heading).abs() < 1e-5);
    } else {
        panic!("car 1 not found");
    }

    if let Some(p) = car_positions[1] {
        assert_eq!(p.x, 645.0);
        assert_eq!(p.y, 557.0);
        let expected_heading = 0.2950021891952857;
        assert!((p.heading - expected_heading).abs() < 1e-5);
    } else {
        panic!("car 2 not found");
    }
}

fn open_images(frame_path: &str) -> (u32, u32, Vec<u8>, Vec<u8>, Vec<u8>) {
    let frame = image::open(frame_path).unwrap().to_luma();

    let bg = image::open("static_test_bg.tiff")
        .unwrap()
        .to_luma()
        .into_raw();

    let track_mask = image::open("static_test_track_mask.tiff")
        .unwrap()
        .to_luma()
        .into_raw();

    let width = frame.width();
    let height = frame.height();
    let frame = frame.into_raw();

    (width, height, frame, bg, track_mask)
}
