#![allow(non_snake_case)]

#[macro_use]
extern crate log;
extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate control_model;
extern crate osqp_wrapper;
extern crate prelude;
extern crate sparse;
extern crate track;

use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::time::Duration;

use control_model::ControlModel;
use prelude::*;
use track::TrackAndLookup;

mod mpc_base;
use mpc_base::MpcBase;

mod mpc_reference;
pub use mpc_reference::MpcReference;

mod mpc_raceline;
pub use mpc_raceline::MpcRaceline;

mod mpc_min_time;
pub use mpc_min_time::MpcMinTime;

mod osqp_mpc_builder;
pub use osqp_mpc_builder::{MpcStage, OsqpMpc};

pub trait InitController<M: ControlModel>: Controller<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    type Config: DeserializeOwned;

    fn new(model: &M, N: u32, track: &Arc<TrackAndLookup>, config: Self::Config) -> Self;

    // TODO: Make an associated const once stable
    fn name() -> &'static str;
}

pub trait Controller<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
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
