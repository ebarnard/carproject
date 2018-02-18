#![allow(non_snake_case)]

extern crate control_model;
#[macro_use]
extern crate log;
extern crate osqp;
extern crate prelude;
extern crate sparse;

use prelude::*;
use control_model::ControlModel;

mod ekf;
pub use ekf::{JointEKF, EKF};

mod param_least_squares;

pub trait Estimator<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NS>, Vector<M::NP>);

    fn param_covariance(&self) -> Matrix<M::NP, M::NP>;
}

#[derive(Debug)]
pub struct Measurement {
    pub position: (float, float),
    pub heading: float,
}
