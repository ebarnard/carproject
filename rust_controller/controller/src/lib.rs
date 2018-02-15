#![allow(non_snake_case)]

extern crate control_model;
#[macro_use]
extern crate log;
extern crate osqp;
extern crate prelude;
extern crate sparse;
extern crate track;

use prelude::*;
use control_model::ControlModel;

mod mpc_base;
use mpc_base::MpcBase;

mod mpc_position;
pub use mpc_position::MpcPosition;

mod mpc_time;
pub use mpc_time::MpcTime;

mod osqp_mpc_builder;
pub use osqp_mpc_builder::OsqpMpc;

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
