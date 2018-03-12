extern crate car_remote;
extern crate winit;

mod utils;

use std::thread;
use std::time::Duration;
use winit::{ControlFlow, ElementState, Event, VirtualKeyCode, WindowEvent};

fn main() {
    let car = utils::parse_car_from_command_line();

    let mut conn = car_remote::Connection::new();
    conn.on(car);
    thread::sleep(Duration::from_millis(300));

    let mut events_loop = winit::EventsLoop::new();

    let _window = winit::WindowBuilder::new()
        .build(&events_loop)
        .expect("could not create window");

    let mut throttle = 0;
    let mut steering = 0;

    events_loop.run_forever(|event| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => match (input.virtual_keycode, input.state) {
                (Some(VirtualKeyCode::Up), ElementState::Pressed) => {
                    throttle = 127;
                }
                (Some(VirtualKeyCode::Up), ElementState::Released) => {
                    throttle = 0;
                }
                (Some(VirtualKeyCode::Down), ElementState::Pressed) => {
                    throttle = -127;
                }
                (Some(VirtualKeyCode::Down), ElementState::Released) => {
                    throttle = 0;
                }
                (Some(VirtualKeyCode::Left), ElementState::Pressed) => {
                    steering = 127;
                }
                (Some(VirtualKeyCode::Left), ElementState::Released) => {
                    steering = 0;
                }
                (Some(VirtualKeyCode::Right), ElementState::Pressed) => {
                    steering = -127;
                }
                (Some(VirtualKeyCode::Right), ElementState::Released) => {
                    steering = 0;
                }
                _ => (),
            },
            _ => (),
        }

        conn.set(car, throttle, steering);

        ControlFlow::Continue
    });
}