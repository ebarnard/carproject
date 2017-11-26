use prelude::*;

mod mpc_position;
pub use self::mpc_position::MpcPosition;

mod osqp_mpc_builder;
pub use self::osqp_mpc_builder::OsqpMpc;

#[derive(Debug, Default)]
pub struct State {
    pub position: (float, float),
    pub velocity: (float, float),
    pub heading: float,
}

#[derive(Debug, Default)]
pub struct Control {
    pub throttle_position: float,
    pub steering_angle: float,
}

pub trait Controller {
    fn step(&mut self, dt: float, state: &State, params: &[float]) -> (Control, State);
}
