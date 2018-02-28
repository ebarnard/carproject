use std::env;
use std::u8;

pub fn parse_car_from_command_line() -> u8 {
    let mut args = env::args();
    // Skip executable path
    args.next();
    if let Some("--car") = args.next().as_ref().map(|a| a.as_str()) {
        if let Some(car) = args.next().and_then(|c| c.parse().ok()) {
            return car;
        }
    }
    panic!("error. car number must be specified by passing --car <CAR>.");
}
