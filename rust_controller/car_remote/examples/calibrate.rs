// If the car is behaving strangely calibrating might fix it
extern crate car_remote;

mod utils;

use std::io;
use std::thread;
use std::time::Duration;

fn main() {
    let PAUSE_DURATION = Duration::from_millis(500);

    let car = utils::parse_car_from_command_line();

    let stdin = io::stdin();
    let mut buf = String::new();
    let mut conn = car_remote::Connection::new();

    // Turn off the controller
    conn.off(car);

    println!("centre steering trim knob. press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    println!("turn G.SPD L fully counterclockwise (to the left). press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    // Set steering and throttle to full right and full brake position
    conn.set(car, -128, -128);
    thread::sleep(PAUSE_DURATION);

    // Turn on the controller
    conn.on(car);
    thread::sleep(PAUSE_DURATION);

    // Set steering and throttle back to centre position
    conn.set(car, 0, 0);
    thread::sleep(PAUSE_DURATION);

    // Set seering full left, full right, full left, full right, centre
    conn.set(car, 0, 127);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, 0, -128);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, 0, 127);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, 0, -128);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, 0, 0);
    thread::sleep(PAUSE_DURATION);

    // Set throttle full forward, full brake, full forward, full brake, centre
    conn.set(car, 127, 0);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, -128, 0);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, 127, 0);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, -128, 0);
    thread::sleep(PAUSE_DURATION);
    conn.set(car, 0, 0);
    thread::sleep(PAUSE_DURATION);

    println!("turn G.SPD L fully clockwise (to the right). press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    conn.off(car);
    thread::sleep(PAUSE_DURATION);

    println!("calibration complete.")
}
