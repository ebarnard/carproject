extern crate car_remote;

use std::thread;
use std::time::Duration;

fn main() {
    let mut conn = car_remote::Connection::new();
    conn.on(0);
    thread::sleep_ms(300);

    println!("running");

    for i in (-128..127).chain((-128..127).rev()).cycle() {
        println!("write {}", i);
        conn.set(0, i, i);
        thread::sleep_ms(10);
    }
}
