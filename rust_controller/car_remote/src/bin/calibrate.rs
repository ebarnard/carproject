// If the car is behaving strangely calibrating might fix it
extern crate car_remote;

use std::io;
use std::thread;
use std::sync::atomic::{compiler_fence, Ordering};

fn main() {
    let PAUSE_MILLIS = 500;

    let stdin = io::stdin();
    let mut buf = String::new();
    let mut conn = car_remote::Connection::new();

    // Turn off the controller
    conn.off(0);

    println!("centre steering trim knob. press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    println!("turn G.SPD L fully counterclockwise (to the left). press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    compiler_fence(Ordering::SeqCst);

    // Set steering and throttle to full right and full brake position
    conn.set(0, -128, -128);
    thread::sleep_ms(PAUSE_MILLIS);

    // Turn on the controller
    conn.on(0);
    thread::sleep_ms(PAUSE_MILLIS);

    // Set steering and throttle back to centre position
    conn.set(0, 0, 0);
    thread::sleep_ms(PAUSE_MILLIS);

    // Set seering full left, full right, full left, full right, centre
    conn.set(0, 0, 127);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, 0, -128);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, 0, 127);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, 0, -128);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, 0, 0);
    thread::sleep_ms(PAUSE_MILLIS);

    // Set throttle full forward, full brake, full forward, full brake, centre
    conn.set(0, 127, 0);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, -128, 0);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, 127, 0);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, -128, 0);
    thread::sleep_ms(PAUSE_MILLIS);
    conn.set(0, 0, 0);
    thread::sleep_ms(PAUSE_MILLIS);

    println!("turn G.SPD L fully clockwise (to the right). press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    conn.off(0);
    thread::sleep_ms(PAUSE_MILLIS);

    println!("calibration complete")
}
