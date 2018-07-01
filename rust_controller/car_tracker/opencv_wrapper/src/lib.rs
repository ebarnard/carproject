pub fn draw_car_position(data: &mut [u8], width: u32, height: u32, x: f64, y: f64, heading: f64) {
    unsafe {
        assert_eq!(data.len(), width as usize * height as usize);
        ffi::draw_car_position(data.as_mut_ptr(), width, height, x, y, heading);
    }
}

pub fn show_greyscale_image(data: &[u8], width: u32, height: u32, delay: u32) {
    unsafe {
        assert_eq!(data.len(), width as usize * height as usize);
        ffi::show_greyscale_image(data.as_ptr(), width, height, delay);
    }
}

mod ffi {
    extern "C" {
        pub fn draw_car_position(
            data: *mut u8,
            width: u32,
            height: u32,
            x: f64,
            y: f64,
            heading: f64,
        );
        pub fn show_greyscale_image(data: *const u8, width: u32, height: u32, delay: u32);
    }
}
