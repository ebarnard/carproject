use prelude::*;
use controller::{Control, State as ControllerState};

#[derive(Debug, Default)]
pub struct State {
    pub position: (float, float),
    pub velocity: (float, float),
    pub heading: float,
}

impl State {
    pub fn to_controller_state(&self) -> ControllerState {
        ControllerState {
            position: self.position,
            velocity: self.velocity,
            heading: self.heading,
        }
    }
}

pub trait SimulationModel {
    fn step(&mut self, dt: float, state: &State, control: &Control) -> State;
    fn params(&self) -> &[float];
}
