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

    // Turn steering fully to the right
    conn.set(car, 0, -128);
    thread::sleep(PAUSE_DURATION);

    // Turn on the controller
    conn.on(car);
    thread::sleep(PAUSE_DURATION);

    println!("turn on car while holding down pairing button. release pairing button. press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    // Turn off the controller
    conn.off(car);
    thread::sleep(PAUSE_DURATION);

    println!("turn off car. press enter to continue.");
    stdin.read_line(&mut buf).expect("could not read stdin");

    println!("pairing complete.")
}
