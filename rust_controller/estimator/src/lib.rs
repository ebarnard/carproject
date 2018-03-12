#![allow(non_snake_case)]

extern crate control_model;
#[macro_use]
extern crate log;
extern crate osqp;
extern crate prelude;
extern crate sparse;

use nalgebra::U3;

use prelude::*;
use control_model::ControlModel;

mod ekf;
pub use ekf::JointEKF;

mod param_least_squares;

type NM = U3;

pub trait Estimator<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
    ) -> (&Vector<M::NS>, &Vector<M::NP>);

    fn param_covariance(&self) -> Matrix<M::NP, M::NP>;
}

#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub position: (float, float),
    pub heading: float,
}
