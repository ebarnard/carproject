extern crate car_remote;

mod utils;

use std::thread;
use std::time::Duration;

fn main() {
    let car = utils::parse_car_from_command_line();

    let mut conn = car_remote::Connection::new();
    conn.on(car);
    thread::sleep(Duration::from_millis(300));

    println!("running");

    for i in (-128..127).chain((-128..127).rev()).cycle() {
        println!("write {}", i);
        conn.set(car, i, i);
        thread::sleep(Duration::from_millis(10));
    }
}
