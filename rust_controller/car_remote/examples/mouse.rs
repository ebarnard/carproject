extern crate car_remote;
extern crate winit;

mod utils;

use std::cell::Cell;
use std::thread;
use std::time::Duration;
use winit::{ControlFlow, Event, WindowEvent};

fn main() {
    let car = utils::parse_car_from_command_line();

    let mut conn = car_remote::Connection::new();
    conn.on(car);
    thread::sleep(Duration::from_millis(300));

    let mut events_loop = winit::EventsLoop::new();

    let window = winit::WindowBuilder::new()
        .build(&events_loop)
        .expect("could not create window");

    let mut throttle = 0.0f64;
    let mut steering = 0.0f64;

    let (w, h): (f64, f64) = window.get_inner_size().expect("zero sized window").into();
    let (w, h) = (Cell::new(w), Cell::new(h));

    events_loop.run_forever(|event| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                let (x, y): (f64, f64) = position.into();
                throttle = 1.0 - y / (h.get() / 2.0);
                steering = 1.0 - x / (w.get() / 2.0);
            }
            _ => (),
        }

        conn.set(
            car,
            (throttle * 128.0).floor() as i8 / 4,
            (steering * 128.0).floor() as i8,
        );

        ControlFlow::Continue
    });
}
