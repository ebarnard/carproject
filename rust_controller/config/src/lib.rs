#![allow(non_snake_case)]

extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate toml;

use std::io::Read;
use std::fs::File;
pub use toml::Value;

static CONFIG_FILE: &'static str = "config.toml";

#[derive(Deserialize)]
pub struct Config {
    pub t: f64,
    pub dt: f64,
    pub track: String,
    pub controller: Controller,
    pub simulator: Simulator,
}

#[derive(Deserialize)]
pub struct Controller {
    pub model: String,
    pub R: Vec<f64>,
    pub Q_state: Vec<f64>,
    pub Q_initial_params: Vec<f64>,
    pub Q_params_multiplier: f64,
    pub initial_params: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Simulator {
    pub model: String,
    pub params: Vec<f64>,
}

pub fn load() -> Config {
    let mut config_str = String::new();
    File::open(CONFIG_FILE)
        .expect("unable to open config.toml")
        .read_to_string(&mut config_str)
        .expect("unable to read config.toml as utf8");
    toml::from_str(&config_str[..]).expect("unable to deserialise config.toml")
}
