pub type Color = [f32; 4];
pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
pub const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
pub const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
pub const BLUE: [f32; 4] = [0.4, 0.4, 1.0, 1.0];
pub const MAGENTA: [f32; 4] = [0.9, 0.9, 0.0, 1.0];

pub(crate) fn pack_color(color: Color) -> u32 {
    fn sat(val: f32) -> u32 {
        if val < 0.0 {
            0
        } else if val > 1.0 {
            255
        } else {
            (val * 255.0) as u32
        }
    }

    // input: [R G B A]
    // output: A | R | G | B
    sat(color[0]) << 16 | sat(color[1]) << 8 | sat(color[2]) | sat(color[3]) << 24
}
