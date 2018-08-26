#![allow(non_snake_case)]

extern crate control_model;
#[macro_use]
extern crate log;
extern crate osqp_wrapper;
extern crate prelude;
extern crate sparse;

use nalgebra::{DimNameSum, U3};

use control_model::ControlModel;
use prelude::*;

mod ekf;
pub use ekf::JointEkf;

mod ukf;
pub use ukf::JointUkf;

mod param_least_squares;

pub type NM = U3;

pub trait Estimator<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn new(
        initial_params: Vector<M::NP>,
        Q_state: Matrix<M::NS, M::NS>,
        Q_params: Matrix<M::NP, M::NP>,
        Q_initial_params: Matrix<M::NP, M::NP>,
        R: Matrix<NM, NM>,
    ) -> Self
    where
        Self: Sized;

    // TODO: Make an associated const once stable
    fn name() -> &'static str
    where
        Self: Sized;

    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
    ) -> (
        &Vector<M::NS>,
        &Vector<M::NP>,
        &Matrix<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>,
    );
}

#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub position: (float, float),
    pub heading: float,
}
