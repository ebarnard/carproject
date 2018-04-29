use std::fs::File;
use std::io::Read;
use toml;

use prelude::*;

static CONFIG_FILE: &'static str = "simulator.toml";

#[derive(Deserialize)]
pub struct SimulatorConfig {
    pub t: float,
    pub real_time: bool,
    pub model: String,
    pub params: Vec<float>,
    pub R: Vec<float>,
    pub Q_state: Vec<float>,
}

impl SimulatorConfig {
    pub fn load() -> SimulatorConfig {
        let mut config_str = String::new();
        File::open(CONFIG_FILE)
            .expect("unable to open simulator.toml")
            .read_to_string(&mut config_str)
            .expect("unable to read simulator.toml as utf8");
        toml::from_str(&config_str[..]).expect("unable to deserialise simulator.toml")
    }
}
