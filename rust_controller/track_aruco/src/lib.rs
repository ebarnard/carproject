#![allow(non_snake_case)]

extern crate prelude;

use nalgebra::Matrix3;
use std::mem;

use prelude::*;

pub fn find_homography(frame: &[u8], width: u32, height: u32) -> Result<Matrix3<float>, ()> {
    let marker_width = 0.060;
    let markers = &[
        ffi::Marker::new(8, -0.305, -0.490, marker_width),
        ffi::Marker::new(10, -0.902, 0.450, marker_width),
        ffi::Marker::new(11, 0.848, 0.440, marker_width),
        ffi::Marker::new(13, 0.848, -0.498, marker_width),
        ffi::Marker::new(16, 0.298, -0.178, marker_width),
        ffi::Marker::new(14, -0.162, 0.246, marker_width),
    ];

    unsafe {
        let mut H = mem::zeroed();
        let markers_found = ffi::find_homography(
            markers.as_ptr(),
            markers.len() as u32,
            frame.as_ptr(),
            width,
            height,
            1,
            &mut H,
        );
        if markers_found < 4 {
            return Err(());
        }
        Ok(Matrix3::from_row_slice(&H.values))
    }
}

mod ffi {
    #[repr(C)]
    pub struct Marker {
        pub id: u32,
        pub x: f64,
        pub y: f64,
        pub width: f64,
    }

    impl Marker {
        pub fn new(id: u32, x: f64, y: f64, width: f64) -> Marker {
            Marker { id, x, y, width }
        }
    }

    #[repr(C)]
    pub struct Homography2 {
        pub values: [f64; 9],
    }

    extern "C" {
        pub fn find_homography(
            markers: *const Marker,
            num_markers: u32,
            frame: *const u8,
            width: u32,
            height: u32,
            debug_print: i32,
            H: *mut Homography2,
        ) -> u32;
    }
}
