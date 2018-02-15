extern crate serialport;

use std::time::{Duration, Instant};
use std::thread;

// WCH USB to serial chip from Arduino Nano ripoff
const USB_VID: u16 = 6790;
const USB_PID: u16 = 29987;

pub struct Connection {
    port: Box<serialport::SerialPort>,
}

impl Connection {
    pub fn new() -> Connection {
        let port = serialport::available_ports()
            .expect("could not enumerate serial ports")
            .into_iter()
            .filter(|port| {
                if let serialport::SerialPortType::UsbPort(ref usb) = port.port_type {
                    if usb.vid == USB_VID && usb.pid == USB_PID {
                        return true;
                    }
                }
                false
            })
            .next()
            .expect("could not find Arduino Nano serial port");

        let settings = serialport::SerialPortSettings {
            baud_rate: serialport::BaudRate::Baud115200,
            timeout: Duration::from_millis(1),
            ..Default::default()
        };

        let port = serialport::open_with_settings(&port.port_name, &settings)
            .expect("could not open serial port");

        let mut conn = Connection { port };

        // Arduino boards always reset on serial connect
        // TODO: Make const
        let READY_TIMEOUT = Duration::from_secs(2);
        let start_time = Instant::now();

        // Wait for the device to be ready
        let mut message = String::new();
        loop {
            conn.read(&mut message);
            if message.len() > 0 {
                break;
            } else if Instant::now() - start_time > READY_TIMEOUT {
                panic!("board not ready");
            } else {
                thread::yield_now();
            }
        }

        conn
    }

    pub fn on(&mut self, car: u8) {
        write!(self.port, "{} ON\r\n", car).expect("could not write to serial port");
    }

    pub fn off(&mut self, car: u8) {
        write!(self.port, "{} OFF\r\n", car).expect("could not write to serial port");
    }

    pub fn set(&mut self, car: u8, throttle: i8, steering: i8) {
        let throttle = (throttle as i16 + 128) as u8;
        let steering = (steering as i16 + 128) as u8;
        write!(self.port, "{} {} {}\n", car, throttle, steering)
            .expect("could not write to serial port");
    }

    pub fn read(&mut self, str_buf: &mut String) {
        let mut buf = [0; 1024];
        if let Ok(len) = self.port.read(&mut buf[..]) {
            return str_buf.push_str(&String::from_utf8_lossy(&buf[..len]));
        }
    }
}
