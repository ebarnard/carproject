use imgui::{self, ImGuiCond, ImVec2, Ui, WindowDrawList};

use {pack_color, Color};

pub struct Display<'ui> {
    pub(crate) ui: imgui::Ui<'ui>,
}

impl<'ui> Display<'ui> {
    pub fn draw_canvas_window(
        &mut self,
        name: &str,
        default_size: (f64, f64),
        f: &mut FnMut(&mut Canvas),
    ) {
        self.ui
            .window(im_str!("{}", name))
            .size(
                (default_size.0 as f32, default_size.1 as f32),
                ImGuiCond::FirstUseEver,
            ).title_bar(false)
            .collapsible(false)
            .build(|| {
                let mut draw_list = self.ui.get_window_draw_list();
                Canvas::draw(&self.ui, &mut draw_list, f);
            });
    }
}

pub struct Canvas<'a, 'ui: 'a> {
    // x, y, width, height
    viewport: [f32; 4],
    ui: &'a Ui<'ui>,
    draw_list: &'a mut WindowDrawList<'ui>,
}

impl<'a, 'ui: 'a> Canvas<'a, 'ui> {
    pub fn draw(
        ui: &'a Ui<'ui>,
        draw_list: &'a mut WindowDrawList<'ui>,
        f: &mut FnMut(&mut Canvas),
    ) {
        let mut window_pos = ImVec2::new(0.0, 0.0);
        let mut window_size = ImVec2::new(0.0, 0.0);
        unsafe {
            imgui::sys::igGetWindowPos(&mut window_pos);
            imgui::sys::igGetWindowSize(&mut window_size);
        }
        let viewport = [window_pos.x, window_pos.y, window_size.x, window_size.y];

        f(&mut Canvas {
            viewport,
            ui,
            draw_list,
        });
    }

    pub fn size(&self) -> (f64, f64) {
        (self.viewport[2] as f64, self.viewport[3] as f64)
    }

    pub fn subview(&mut self, viewport: [f64; 4], f: &mut FnMut(&mut Canvas)) {
        if viewport[0] < 0.0 || viewport[1] < 0.0 {
            //println!("viewport x and y must be positive");
            return;
        }

        if viewport[2] <= 0.0 || viewport[3] <= 0.0 {
            return;
        }

        let mut canvas = Canvas {
            // Clipping rectangle has (0, 0) in top left corner
            viewport: [
                self.viewport[0] + viewport[0] as f32,
                self.viewport[1] + viewport[1] as f32,
                viewport[2] as f32,
                viewport[3] as f32,
            ],
            ui: self.ui,
            draw_list: self.draw_list,
        };

        f(&mut canvas);
    }

    pub fn line(&mut self, color: Color, radius: f64, line: [f64; 4]) {
        self.draw_list
            .add_line(
                (
                    self.viewport[0] + line[0] as f32,
                    self.viewport[1] + line[1] as f32,
                ),
                (
                    self.viewport[0] + line[2] as f32,
                    self.viewport[1] + line[3] as f32,
                ),
                pack_color(color),
            ).thickness(radius as f32)
            .build();
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

        unsafe {
            let start = text.as_ptr();
            let end = start.offset(text.len() as isize);
            imgui::sys::ImDrawList_AddText(
                imgui::sys::igGetWindowDrawList(),
                ImVec2::new(self.viewport[0] + x_pos, self.viewport[1] + y_pos),
                pack_color(color),
                start as *const i8,
                end as *const i8,
            );
        }
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
