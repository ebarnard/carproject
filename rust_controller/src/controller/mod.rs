use prelude::*;
use control_model::ControlModel;

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

pub trait Controller<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NI>, Vector<M::NS>);
}
