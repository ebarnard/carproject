use std::io::Read;
use std::fs::File;
use toml;

use prelude::*;

static CONFIG_FILE: &'static str = "controller.toml";

#[derive(Deserialize)]
pub struct ControllerConfig {
    pub dt: float,
    pub track: String,
    pub N: u32,
    pub model: String,
    pub controller: String,
    pub R: Vec<float>,
    pub Q_state: Vec<float>,
    pub Q_initial_params: Vec<float>,
    pub Q_params_multiplier: float,
    pub initial_params: Vec<float>,
    pub u_min: Vec<float>,
    pub u_max: Vec<float>,
}

impl ControllerConfig {
    pub fn load() -> ControllerConfig {
        let mut config_str = String::new();
        File::open(CONFIG_FILE)
            .expect("unable to open controller.toml")
            .read_to_string(&mut config_str)
            .expect("unable to read controller.toml as utf8");
        toml::from_str(&config_str[..]).expect("unable to deserialise controller.toml")
    }
}
