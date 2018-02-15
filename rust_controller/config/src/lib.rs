#![allow(non_snake_case)]

extern crate prelude;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate toml;

use prelude::*;
use std::io::Read;
use std::fs::File;
pub use toml::Value;

static CONFIG_FILE: &'static str = "config.toml";

#[derive(Deserialize)]
pub struct Config {
    pub t: float,
    pub dt: float,
    pub track: String,
    pub controller: Controller,
    pub simulator: Simulator,
}

#[derive(Deserialize)]
pub struct Controller {
    pub N: u32,
    pub model: String,
    pub R: Vec<float>,
    pub Q_state: Vec<float>,
    pub Q_initial_params: Vec<float>,
    pub Q_params_multiplier: float,
    pub initial_params: Vec<float>,
}

#[derive(Deserialize)]
pub struct Simulator {
    pub real_time: bool,
    pub model: String,
    pub params: Vec<float>,
}

pub fn load() -> Config {
    let mut config_str = String::new();
    File::open(CONFIG_FILE)
        .expect("unable to open config.toml")
        .read_to_string(&mut config_str)
        .expect("unable to read config.toml as utf8");
    toml::from_str(&config_str[..]).expect("unable to deserialise config.toml")
}
