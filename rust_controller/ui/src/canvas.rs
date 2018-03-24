use imgui::{self, ImVec2, Ui, WindowDrawList};

use {pack_color, Color};

pub struct Canvas<'a, 'ui: 'a> {
    // x, y, width, height
    viewport: [u32; 4],
    // Ratio of framebuffer resolution to window size. 1 normally, 2 for a retina display.
    hidpi_factor: f64,
    //ui: &'a Ui<'a>,
    draw_list: &'a mut WindowDrawList<'ui>,
}

impl<'a, 'ui: 'a> Canvas<'a, 'ui> {
    pub fn draw(viewport: [u32; 4], ui: &Ui, f: &mut FnMut(&mut Canvas)) {
        ui.with_window_draw_list(|draw_list| {
            f(&mut Canvas {
                viewport,
                hidpi_factor: 1.0,
                //ui,
                draw_list,
            });
        });
    }

    pub fn size(&self) -> (f64, f64) {
        (
            self.viewport[2] as f64 / self.hidpi_factor,
            self.viewport[3] as f64 / self.hidpi_factor,
        )
    }

    pub fn subview(&mut self, viewport: [f64; 4], f: &mut FnMut(&mut Canvas)) {
        assert!(
            viewport.iter().all(|&v| v >= 0.0),
            "all viewport values must be positive"
        );

        /*assert!(
            self.viewport[2] as f64 >= (viewport[0] + viewport[2]) * self.hidpi_factor,
            "subview cannot extend outside current viewport in x direction"
        );
        assert!(
            self.viewport[1] as f64 >= (viewport[1] + viewport[3]) * self.hidpi_factor,
            "subview cannot extend outside current viewport in y direction"
        );*/

        let mut canvas = Canvas {
            // Clipping rectangle has (0, 0) in bottom left corner
            viewport: [
                self.viewport[0] + (viewport[0] * self.hidpi_factor).floor() as u32,
                self.viewport[1] + (viewport[1] * self.hidpi_factor).floor() as u32,
                (viewport[2] * self.hidpi_factor).floor() as u32,
                (viewport[3] * self.hidpi_factor).floor() as u32,
            ],
            hidpi_factor: self.hidpi_factor,
            //ui: &*self.ui,
            draw_list: self.draw_list,
        };

        f(&mut canvas);
    }

    pub fn line(&mut self, color: Color, radius: f64, line: [f64; 4]) {
        self.draw_list.add_line(
            ImVec2::new(
                self.viewport[0] as f32 + line[0] as f32,
                self.viewport[1] as f32 + line[1] as f32,
            ),
            ImVec2::new(
                self.viewport[0] as f32 + line[2] as f32,
                self.viewport[1] as f32 + line[3] as f32,
            ),
            pack_color(color),
            radius as f32,
        );
    }

    pub fn text(
        &mut self,
        color: Color,
        // TODO: support multiple font sizes with imgui
        _font_size: u32,
        x_align: TextXAlign,
        y_align: TextYAlign,
        text: &str,
        pos: [f64; 2],
    ) {
        let text_size = calc_text_size(text, 1e9);

        let x_pos = match x_align {
            TextXAlign::Left => pos[0] as f32,
            TextXAlign::Right => pos[0] as f32 - text_size.x,
            TextXAlign::Centre => pos[0] as f32 - 0.5 * text_size.x,
        };

        let y_pos = match y_align {
            TextYAlign::Top => pos[1] as f32,
            TextYAlign::Bottom => pos[1] as f32 - text_size.y,
            TextYAlign::Middle => pos[1] as f32 - 0.5 * text_size.y,
        };

        self.draw_list.add_text(
            ImVec2::new(
                self.viewport[0] as f32 + x_pos,
                self.viewport[1] as f32 + y_pos,
            ),
            pack_color(color),
            text,
        );
    }
}

pub enum TextXAlign {
    Left,
    Right,
    Centre,
}

pub enum TextYAlign {
    Top,
    Bottom,
    Middle,
}

fn calc_text_size(text: &str, wrap_width: f32) -> ImVec2 {
    let hide_text_after_double_hash = false;

    let mut buffer = ImVec2::new(0.0, 0.0);
    unsafe {
        let start = text.as_ptr();
        let end = start.offset(text.len() as isize);

        imgui::sys::igCalcTextSize(
            &mut buffer as *mut ImVec2,
            start as *const i8,
            end as *const i8,
            hide_text_after_double_hash,
            wrap_width,
        );
    }
    buffer
}
