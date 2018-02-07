use graphics::{self, Context, Graphics, ImageSize, Transformed};
use graphics::glyph_cache::rusttype::GlyphCache;
use graphics::types::Color;
use glium_graphics::CreateTexture;

use colors::WHITE;

pub trait Canvas {
    fn size(&self) -> (f64, f64);
    fn subview(&mut self, position: [f64; 4], f: &mut FnMut(&mut Canvas));
    fn line(&mut self, color: Color, radius: f64, line: [f64; 4]);
    fn text(&mut self, color: Color, font_size: u32, align: TextAlign, text: &str, pos: [f64; 2]);
}

pub struct GraphicsCanvas<'a, G: 'a + Graphics, F: 'a> {
    // x, y, width, height
    viewport: [u32; 4],
    // Ratio of framebuffer resolution to window size. 1 normally, 2 for a retina display.
    hidpi_factor: f64,
    graphics: &'a mut G,
    context: Context,
    glyph_cache: &'a mut GlyphCache<'static, F, <G as Graphics>::Texture>,
}

impl<'a, G: 'a + Graphics, F: 'a> GraphicsCanvas<'a, G, F>
where
    G::Texture: CreateTexture<F> + ImageSize,
{
    pub fn new(
        g: &'a mut G,
        c: Context,
        gc: &'a mut GlyphCache<'static, F, <G as Graphics>::Texture>,
    ) -> GraphicsCanvas<'a, G, F> {
        let viewport = c.viewport.expect("viewport information required");
        assert!(
            viewport.rect.iter().all(|&v| v >= 0),
            "viewport must not contain negative values"
        );

        // Clearing only once after a resize sometimes leaves an uncleared black area
        graphics::clear(WHITE, g);

        GraphicsCanvas {
            viewport: [
                viewport.rect[0] as u32,
                viewport.rect[1] as u32,
                viewport.rect[2] as u32,
                viewport.rect[3] as u32,
            ],
            hidpi_factor: (viewport.draw_size[0] as f64) / (viewport.window_size[0] as f64),
            graphics: g,
            context: c,
            glyph_cache: gc,
        }
    }
}

impl<'a, G: 'a + Graphics, F: 'a> Canvas for GraphicsCanvas<'a, G, F>
where
    G::Texture: CreateTexture<F> + ImageSize,
{
    fn size(&self) -> (f64, f64) {
        (
            self.viewport[2] as f64 / self.hidpi_factor,
            self.viewport[3] as f64 / self.hidpi_factor,
        )
    }

    fn subview(&mut self, viewport: [f64; 4], f: &mut FnMut(&mut Canvas)) {
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

        let mut canvas = GraphicsCanvas {
            // Clipping rectangle has (0, 0) in bottom left corner
            viewport: [
                self.viewport[0] + (viewport[0] * self.hidpi_factor).floor() as u32,
                self.viewport[1] + self.viewport[3]
                    - ((viewport[1] + viewport[3]) * self.hidpi_factor).floor() as u32,
                (viewport[2] * self.hidpi_factor).floor() as u32,
                (viewport[3] * self.hidpi_factor).floor() as u32,
            ],
            hidpi_factor: self.hidpi_factor,
            graphics: self.graphics,
            context: self.context.trans(viewport[0], viewport[1]),
            glyph_cache: self.glyph_cache,
        };

        f(&mut canvas);
    }

    fn line(&mut self, color: Color, radius: f64, line: [f64; 4]) {
        let draw_state = graphics::DrawState::default().scissor(self.viewport);
        graphics::Line::new(color, radius).draw(
            line,
            &draw_state,
            self.context.transform,
            self.graphics,
        );
    }

    fn text(&mut self, color: Color, font_size: u32, align: TextAlign, text: &str, pos: [f64; 2]) {
        // Scale by 1/hidpi and increase font size by hidpi so the font does not appear blurry
        let font_size = (font_size as f64 * self.hidpi_factor).floor() as u32;

        let pos = {
            let mut text_width = || {
                self.glyph_cache
                    .preload_chars(font_size, text.chars())
                    .map_err(|_| ())
                    .expect("could not preload characters");
                // convert points to pixels
                // TODO: Send a PR about this as it's insane
                let font_px = ((font_size as f32) * 1.333).round() as u32;
                text.chars()
                    .map(|c| {
                        self.glyph_cache
                            .opt_character(font_px, c)
                            .expect("character not loaded")
                            .width()
                    })
                    .sum::<f64>() / self.hidpi_factor
            };

            match align {
                TextAlign::Left => pos,
                TextAlign::Right => [pos[0] - text_width(), pos[1]],
                TextAlign::Centre => [pos[0] - 0.5 * text_width(), pos[1]],
            }
        };

        let draw_state = graphics::DrawState::default().scissor(self.viewport);
        let context = self.context
            .trans(pos[0], pos[1])
            .scale(1.0 / self.hidpi_factor, 1.0 / self.hidpi_factor);
        let _ = graphics::Text::new_color(color, font_size)
            .round()
            .draw(
                text,
                self.glyph_cache,
                &draw_state,
                context.transform,
                self.graphics,
            )
            .map_err(|_| ())
            .expect("could not draw text");
    }
}

pub enum TextAlign {
    Left,
    Right,
    Centre,
}
