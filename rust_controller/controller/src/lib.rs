#![allow(non_snake_case)]

extern crate control_model;
#[macro_use]
extern crate log;
extern crate osqp;
extern crate prelude;
extern crate sparse;
extern crate track;

use std::sync::Arc;
use std::time::Duration;

use control_model::ControlModel;
use prelude::*;
use track::TrackAndLookup;

mod mpc_base;
use mpc_base::MpcBase;

mod mpc_position;
pub use mpc_position::MpcPosition;

mod mpc_distance;
pub use mpc_distance::MpcDistance;

mod osqp_mpc_builder;
pub use osqp_mpc_builder::{MpcStage, OsqpMpc};

pub trait Controller<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn new(model: &M, N: u32, track: &Arc<TrackAndLookup>) -> Self
    where
        Self: Sized;

    // TODO: Make an associated const once stable
    fn name() -> &'static str
    where
        Self: Sized;

    fn update_input_bounds(&mut self, u_min: Vector<M::NI>, u_max: Vector<M::NI>);

    /// `u` has N + 1 columns where the first is the previously applied input.
    fn step(
        &mut self,
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        u: &Matrix<M::NI, Dy>,
        p: &Vector<M::NP>,
        time_limit: Duration,
    ) -> (&Matrix<M::NI, Dy>, &Matrix<M::NS, Dy>);
}
