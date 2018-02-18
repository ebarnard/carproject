extern crate car_remote;
extern crate winit;

use std::cell::Cell;
use winit::{ControlFlow, Event, WindowEvent};

fn main() {
    let mut conn = car_remote::Connection::new();
    let mut events_loop = winit::EventsLoop::new();

    let window = winit::WindowBuilder::new()
        .build(&events_loop)
        .expect("could not create window");

    let mut throttle = 0.0;
    let mut steering = 0.0;

    let (w, h) = window.get_inner_size().expect("zero sized window");
    let (w, h) = (Cell::new(w), Cell::new(h));
    let hidpi_factor = window.hidpi_factor() as f64;

    events_loop.run_forever(|event| {
        match event {
            Event::WindowEvent {
                event:
                    WindowEvent::MouseMoved {
                        position: (x, y), ..
                    },
                ..
            } => {
                throttle = 1.0 - (y as f64) / (hidpi_factor * h.get() as f64 / 2.0);
                steering = 1.0 - (x as f64) / (hidpi_factor * w.get() as f64 / 2.0);
            }
            _ => (),
        }

        conn.set(
            0,
            (throttle * 128.0).floor() as i8 / 4,
            (steering * 128.0).floor() as i8,
        );

        ControlFlow::Continue
    });
}
