use prelude::*;
use control_model::ControlModel;

mod mpc_base;
use self::mpc_base::MpcBase;

mod mpc_position;
pub use self::mpc_position::MpcPosition;

mod mpc_time;
pub use self::mpc_time::MpcTime;

mod osqp_mpc_builder;
pub use self::osqp_mpc_builder::OsqpMpc;

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
    ) -> (&Matrix<M::NI, Dy>, &Matrix<M::NS, Dy>);
}
