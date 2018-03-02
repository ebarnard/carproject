extern crate glium;
extern crate glium_graphics;
extern crate graphics;

use glium::Surface;
use glium::glutin::{self, Event, WindowEvent};

use graphics::Viewport;
pub use graphics::types::Color;

use glium_graphics::{GlyphCache, TextureSettings};

use std::cell::Cell;
use std::sync::{mpsc, Arc};
use std::time::Duration;

mod canvas;
pub use canvas::{Canvas, TextAlign};
pub mod colors;
pub mod plot;

const OPEN_SANS_BYTES: &'static [u8] = include_bytes!("../fonts/open-sans/OpenSans-Regular.ttf");

pub fn min<T: Copy + PartialOrd>(a: T, b: T) -> T {
    if a.lt(&b) {
        a
    } else {
        b
    }
}

pub fn max<T: Copy + PartialOrd>(a: T, b: T) -> T {
    if a.gt(&b) {
        a
    } else {
        b
    }
}

pub trait State: 'static {
    type Event: Send;
    fn update(&mut self, event: Self::Event);
    fn draw(&self, canvas: &mut Canvas);
}

struct StateEraser<T: State> {
    rx: mpsc::Receiver<T::Event>,
    state: T,
}

trait ErasedState {
    fn update(&mut self);
    fn draw(&self, canvas: &mut Canvas);
}

impl<T: State> ErasedState for StateEraser<T> {
    fn update(&mut self) {
        for event in self.rx.try_iter() {
            self.state.update(event);
        }
    }

    fn draw(&self, canvas: &mut Canvas) {
        self.state.draw(canvas)
    }
}

pub struct Window {
    event_loop: glutin::EventsLoop,
    state: Box<ErasedState>,
}

impl Window {
    pub fn new<T: State>(state: T) -> (Window, EventSender<T::Event>) {
        let mut event_loop = glutin::EventsLoop::new();
        // Poll the event loop so we don't race with RecordSender to be the first to use the event
        // loop.
        event_loop.poll_events(|_| ());

        let (tx, rx) = mpsc::channel();

        let event_sender = EventSender {
            tx,
            proxy: Arc::new(event_loop.create_proxy()),
        };

        let winow = Window {
            event_loop,
            state: Box::new(StateEraser { rx, state }),
        };

        (winow, event_sender)
    }

    pub fn run_on_main_thread(&mut self) {
        let window = glutin::WindowBuilder::new()
            .with_title("Hello world!")
            .with_dimensions(1024, 768);
        let context = glutin::ContextBuilder::new().with_gl(glutin::GL_CORE);

        let display = glium::Display::new(window, context, &self.event_loop).unwrap();
        let mut glium_2d = glium_graphics::Glium2d::new(glium_graphics::OpenGL::V3_2, &display);
        let mut open_sans_glyph_cache =
            GlyphCache::from_bytes(OPEN_SANS_BYTES, display.clone(), TextureSettings::new())
                .unwrap();

        let exit = Cell::new(false);
        let render = Cell::new(false);
        let new_records = Cell::new(false);

        let hidpi_factor = Cell::new(display.gl_window().hidpi_factor());

        let handle_event = |event| match event {
            Event::Awakened => {
                new_records.set(true);
                render.set(true);
            }
            Event::WindowEvent {
                event: WindowEvent::Closed,
                ..
            } => {
                exit.set(true);
            }
            Event::WindowEvent {
                event: WindowEvent::Refresh,
                ..
            } => {
                render.set(true);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_, _),
                ..
            } => {
                render.set(true);
            }
            // TODO: Add this back in when glutin updates
            /*Event::WindowEvent { event: WindowEvent::HiDPIFactorChanged(f), .. } => {
                hidpi_factor.set(f);
                render.set(true);
            }*/
            _ => (),
        };

        'running: loop {
            self.event_loop.poll_events(|event| handle_event(event));

            if exit.replace(false) {
                break 'running;
            }

            if new_records.replace(false) {
                // Handle any new events
                self.state.update();
            }

            if render.replace(false) {
                let mut frame = display.draw();

                let (draw_w, draw_h) = frame.get_dimensions();
                let hidpi_factor = hidpi_factor.get();
                let viewport = Viewport {
                    rect: [0, 0, draw_w as i32, draw_h as i32],
                    draw_size: [draw_w, draw_h],
                    window_size: [
                        ((draw_w as f32) / hidpi_factor).floor() as u32,
                        ((draw_h as f32) / hidpi_factor).floor() as u32,
                    ],
                };

                glium_2d.draw(&mut frame, viewport, |c, g| {
                    self.state.draw(&mut canvas::GraphicsCanvas::new(
                        g,
                        c,
                        &mut open_sans_glyph_cache,
                    ));
                });

                frame.finish().expect("failed to draw frame");
            }

            ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));

            self.event_loop.run_forever(|event| {
                handle_event(event);
                glutin::ControlFlow::Break
            });
        }
    }
}

pub struct EventSender<T> {
    tx: mpsc::Sender<T>,
    // TODO: Remove this Arc once we update glium
    proxy: Arc<glutin::EventsLoopProxy>,
}

impl<T> Clone for EventSender<T> {
    fn clone(&self) -> Self {
        EventSender {
            tx: self.tx.clone(),
            proxy: self.proxy.clone(),
        }
    }
}

impl<T> EventSender<T> {
    pub fn send(&mut self, event: T) -> Result<(), ()> {
        self.tx.send(event).map_err(|_| ())?;
        self.proxy.wakeup().map_err(|_| ())?;
        Ok(())
    }
}

pub fn pad_all<F: FnMut(&mut Canvas)>(x: f64, canvas: &mut Canvas, mut f: F) {
    let (w, h) = canvas.size();
    canvas.subview([x, x, w - 2.0 * x, h - 2.0 * x], &mut f);
}
