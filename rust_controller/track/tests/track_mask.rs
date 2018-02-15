#![allow(non_snake_case)]

extern crate prelude;
extern crate track;

use nalgebra::Matrix3;
use prelude::*;
use std::fs::File;
use std::io::Read;

#[test]
fn main() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let H = Matrix3::new(
        5.91267741e+02, -2.36222242e-01, 5.71960743e+02,
        2.53417086e+00, -5.92389039e+02, 5.04326362e+02,
        1.64211802e-03, -5.76778274e-03, 1.00000000e+00,
    );

    let track = track::Track::load("tests/office_desk_track_2500.csv");
    let mask = track.bitmap_mask(&H, 1280, 1024);

    let mut expected_mask = Vec::new();
    File::open("tests/office_desk_track_2500_mask.dat")
        .unwrap()
        .read_to_end(&mut expected_mask)
        .unwrap();

    assert_eq!(mask, expected_mask);

    // Output PNG for visual comparison
    //extern crate cv;
    //use std::io::Write;
    //let mask_mat = cv::Mat::from_buffer(1024 as i32, 1280 as i32, 0, &mask);
    //let bytes = mask_mat.imencode("mask.png", Vec::new()).unwrap();
    //File::create("tests/office_desk_track_2500_mask.png").unwrap().write_all(&bytes).unwrap();
}
