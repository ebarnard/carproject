#![allow(non_snake_case)]

extern crate prelude;

use nalgebra::MatrixMN;
use prelude::*;

mod directvelocity;
pub use directvelocity::DirectVelocity;

mod spenglergammeterbicycle;
pub use spenglergammeterbicycle::SpenglerGammeterBicycle;

mod combine_state;
pub use combine_state::CombineState;

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

pub trait ControlModel
where
    DefaultAllocator: Dims3<Self::NS, Self::NI, Self::NP>,
{
    type NS: DimName;
    type NI: DimName;
    type NP: DimName;

    fn new() -> Self
    where
        Self: Sized;

    fn step(
        &self,
        dt: float,
        x: &Vector<Self::NS>,
        u: &Vector<Self::NI>,
        p: &Vector<Self::NP>,
    ) -> Vector<Self::NS> {
        rk4(dt, 5, x, |x| self.state_equation(x, u, p))
    }

    // Returns the state space derivative at a given operating point
    fn state_equation(
        &self,
        x: &Vector<Self::NS>,
        u: &Vector<Self::NI>,
        p: &Vector<Self::NP>,
    ) -> Vector<Self::NS>;

    // Returns the jacobian of the state space system with respect to its state and inputs
    fn linearise(
        &self,
        x0: &Vector<Self::NS>,
        u0: &Vector<Self::NI>,
        p0: &Vector<Self::NP>,
    ) -> (Matrix<Self::NS, Self::NS>, Matrix<Self::NS, Self::NI>);

    /// Returns two boolean matrices with true everywhere A and B could contain a non-zero value
    fn linearise_sparsity(
        &self,
    ) -> (
        MatrixMN<bool, Self::NS, Self::NS>,
        MatrixMN<bool, Self::NS, Self::NI>,
    ) {
        let A_mask = MatrixMN::<bool, Self::NS, Self::NS>::from_element(true);
        let B_mask = MatrixMN::<bool, Self::NS, Self::NI>::from_element(true);
        (A_mask, B_mask)
    }

    // Returns the jacobian of the state space system with respect to its parameters
    fn linearise_parameters(
        &self,
        x0: &Vector<Self::NS>,
        u0: &Vector<Self::NI>,
        p0: &Vector<Self::NP>,
    ) -> Matrix<Self::NS, Self::NP>;

    fn linearise_parameters_sparsity(&self) -> MatrixMN<bool, Self::NS, Self::NP> {
        MatrixMN::<bool, Self::NS, Self::NP>::from_element(true)
    }

    fn x_from_state(&self, state: &State) -> Vector<Self::NS>;

    fn x_to_state(&self, x: &Vector<Self::NS>) -> State;

    fn u_to_control(&self, u: &Vector<Self::NI>) -> Control {
        Control {
            throttle_position: u[0],
            steering_angle: u[1],
        }
    }

    fn u_from_control(&self, control: &Control) -> Vector<Self::NI> {
        let mut u: Vector<Self::NI> = nalgebra::zero();
        u[0] = control.throttle_position;
        u[1] = control.steering_angle;
        u
    }

    /// Returns the mininum and maximum allowable input values.
    fn input_bounds(&self) -> (Vector<Self::NI>, Vector<Self::NI>);

    /// Returns the mininum and maximum allowable input change at each mpc iteration.
    /// Linearised models can be wildly incorrect with large deltas.
    fn input_delta_bounds(&self) -> (Vector<Self::NI>, Vector<Self::NI>);
}

pub fn discretise<NS: DimName, NI: DimName>(
    dt: float,
    A: &Matrix<NS, NS>,
    B: &Matrix<NS, NI>,
) -> (Matrix<NS, NS>, Matrix<NS, NI>)
where
    DefaultAllocator: Dims2<NS, NI>,
{
    // Second order taylor approximation for matrix exponential exp(x)
    let I = Matrix::<NS, NS>::identity();

    let A_d = &I + (A * dt) + (A * A * dt * dt / 2.0);
    let B_d = ((I * dt) + (A * dt * dt / 2.0)) * B;

    (A_d, B_d)
}

pub fn discretise_sparsity<NS: DimName, NI: DimName>(
    A: &MatrixMN<bool, NS, NS>,
    B: &MatrixMN<bool, NS, NI>,
) -> (MatrixMN<bool, NS, NS>, MatrixMN<bool, NS, NI>)
where
    DefaultAllocator: Dims2<NS, NI>,
{
    let A = A.map(|_| 1.0);
    let B = B.map(|_| 1.0);

    let (A_d, B_d) = discretise(1.0, &A, &B);

    (A_d.map(|v| v != 0.0), B_d.map(|v| v != 0.0))
}

fn rk4<N: DimName, F>(dt: float, num_steps: u32, y_0: &Vector<N>, mut f: F) -> Vector<N>
where
    F: FnMut(&Vector<N>) -> Vector<N>,
    DefaultAllocator: Allocator<float, N>,
{
    let h = dt / float::from(num_steps);
    let mut y = y_0.clone();
    for _ in 0..num_steps {
        let k1 = f(&y) * h;
        let k2 = f(&(&y + 0.5 * &k1)) * h;
        let k3 = f(&(&y + 0.5 * &k2)) * h;
        let k4 = f(&(&y + &k3)) * h;
        y += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
    }
    y
}
