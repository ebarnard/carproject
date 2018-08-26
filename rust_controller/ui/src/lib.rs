extern crate gl;
extern crate glutin;
#[macro_use]
extern crate imgui;
extern crate imgui_opengl_renderer;

use glutin::{
    ElementState, Event, GlContext, MouseButton, MouseScrollDelta, TouchPhase,
    VirtualKeyCode as Key, WindowEvent,
};
use imgui::{FrameSize, ImGui};
use std::cell::Cell;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

mod canvas;
pub mod colors;
pub mod plot;

pub use canvas::{Canvas, Display, TextXAlign, TextYAlign};
use colors::pack_color;
pub use colors::Color;

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
    fn draw(&self, display: &mut Display);
}

struct StateEraser<T: State> {
    rx: mpsc::Receiver<T::Event>,
    state: T,
}

trait ErasedState {
    fn update(&mut self);
    fn draw(&self, display: &mut Display);
}

impl<T: State> ErasedState for StateEraser<T> {
    fn update(&mut self) {
        for event in self.rx.try_iter() {
            self.state.update(event);
        }
    }

    fn draw(&self, display: &mut Display) {
        self.state.draw(display)
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
            .with_title("Car")
            .with_dimensions((1024, 768).into());
        let context = glutin::ContextBuilder::new().with_gl(glutin::GL_CORE);

        let gl_window = glutin::GlWindow::new(window, context, &self.event_loop).unwrap();
        unsafe { gl_window.make_current().unwrap() };
        gl::load_with(|s| gl_window.get_proc_address(s) as _);

        let mut imgui = ImGui::init();
        imgui.set_ini_filename(None);
        let renderer = imgui_opengl_renderer::Renderer::new(&mut imgui, |s| {
            gl_window.get_proc_address(s) as _
        });

        let exit = Cell::new(false);
        let render = Cell::new(false);
        let new_records = Cell::new(false);

        let mouse_pos = Cell::new((0.0, 0.0));
        let mouse_left_pressed = Cell::new(false);
        let mouse_right_pressed = Cell::new(false);
        let mouse_middle_pressed = Cell::new(false);
        let mouse_scroll_y = Cell::new(0.0);

        let mut last_frame = Instant::now();

        let handle_event = |imgui: &mut ImGui, event| {
            let window_event = match event {
                Event::Awakened => {
                    new_records.set(true);
                    render.set(true);
                    return;
                }
                Event::WindowEvent { event, .. } => event,
                _ => return,
            };

            match window_event {
                WindowEvent::CloseRequested => {
                    exit.set(true);
                }
                WindowEvent::Refresh => {
                    render.set(true);
                }
                WindowEvent::Resized(_) => {
                    render.set(true);
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    let pressed = input.state == ElementState::Pressed;
                    match input.virtual_keycode {
                        Some(Key::Tab) => imgui.set_key(0, pressed),
                        Some(Key::Left) => imgui.set_key(1, pressed),
                        Some(Key::Right) => imgui.set_key(2, pressed),
                        Some(Key::Up) => imgui.set_key(3, pressed),
                        Some(Key::Down) => imgui.set_key(4, pressed),
                        Some(Key::PageUp) => imgui.set_key(5, pressed),
                        Some(Key::PageDown) => imgui.set_key(6, pressed),
                        Some(Key::Home) => imgui.set_key(7, pressed),
                        Some(Key::End) => imgui.set_key(8, pressed),
                        Some(Key::Delete) => imgui.set_key(9, pressed),
                        Some(Key::Back) => imgui.set_key(10, pressed),
                        Some(Key::Return) => imgui.set_key(11, pressed),
                        Some(Key::Escape) => imgui.set_key(12, pressed),
                        Some(Key::A) => imgui.set_key(13, pressed),
                        Some(Key::C) => imgui.set_key(14, pressed),
                        Some(Key::V) => imgui.set_key(15, pressed),
                        Some(Key::X) => imgui.set_key(16, pressed),
                        Some(Key::Y) => imgui.set_key(17, pressed),
                        Some(Key::Z) => imgui.set_key(18, pressed),
                        Some(Key::LControl) | Some(Key::RControl) => imgui.set_key_ctrl(pressed),
                        Some(Key::LShift) | Some(Key::RShift) => imgui.set_key_shift(pressed),
                        Some(Key::LAlt) | Some(Key::RAlt) => imgui.set_key_alt(pressed),
                        Some(Key::LWin) | Some(Key::RWin) => imgui.set_key_super(pressed),
                        _ => {}
                    }
                    render.set(true);
                }
                WindowEvent::CursorMoved {
                    position: logical_position,
                    ..
                } => {
                    let (x, y): (f64, f64) = logical_position.into();
                    mouse_pos.set((x, y));
                    render.set(true);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    match button {
                        MouseButton::Left => mouse_left_pressed.set(state == ElementState::Pressed),
                        MouseButton::Right => {
                            mouse_right_pressed.set(state == ElementState::Pressed)
                        }
                        MouseButton::Middle => {
                            mouse_middle_pressed.set(state == ElementState::Pressed)
                        }
                        _ => return,
                    }
                    render.set(true);
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, y_lines),
                    phase: TouchPhase::Moved,
                    ..
                } => {
                    // TODO: Figure out line height and scroll by y_lines * line_height logical
                    // pixels.
                    mouse_scroll_y.set(mouse_scroll_y.get() + y_lines);
                    render.set(true);
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::PixelDelta(logical_delta),
                    phase: TouchPhase::Moved,
                    ..
                } => {
                    mouse_scroll_y.set(mouse_scroll_y.get() + logical_delta.y as f32);
                    render.set(true);
                }
                WindowEvent::ReceivedCharacter(c) => {
                    imgui.add_input_character(c);
                    render.set(true);
                }
                _ => (),
            }
        };

        'running: loop {
            self.event_loop
                .poll_events(|event| handle_event(&mut imgui, event));

            if exit.replace(false) {
                break 'running;
            }

            if new_records.replace(false) {
                // Handle any new events
                self.state.update();
            }

            if render.replace(false) {
                let scale = imgui.display_framebuffer_scale();
                let mouse_pos = mouse_pos.get();
                imgui.set_mouse_pos(mouse_pos.0 as f32 / scale.0, mouse_pos.1 as f32 / scale.1);
                imgui.set_mouse_down([
                    mouse_left_pressed.get(),
                    mouse_right_pressed.get(),
                    mouse_middle_pressed.get(),
                    false,
                    false,
                ]);
                imgui.set_mouse_wheel(mouse_scroll_y.replace(0.0) / scale.1);

                let logical_size: (f64, f64) = gl_window.get_inner_size().unwrap().into();
                let hidpi_factor = gl_window.get_hidpi_factor();

                let now = Instant::now();
                let delta = now - last_frame;
                let delta_s =
                    delta.as_secs() as f32 + delta.subsec_nanos() as f32 / 1_000_000_000.0;
                last_frame = now;

                let ui = imgui.frame(
                    FrameSize::new(logical_size.0, logical_size.1, hidpi_factor),
                    delta_s,
                );

                let mut canvas_display = canvas::Display { ui };
                self.state.draw(&mut canvas_display);

                unsafe {
                    gl::ClearColor(1.0, 1.0, 1.0, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT);
                }
                renderer.render(canvas_display.ui);
                gl_window.swap_buffers().unwrap();
            }

            thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));

            self.event_loop.run_forever(|event| {
                handle_event(&mut imgui, event);
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
