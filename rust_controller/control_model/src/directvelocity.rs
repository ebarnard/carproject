use nalgebra::{self, Matrix3, Matrix3x2, Vector2, Vector3};
use nalgebra::dimension::{U0, U2, U3};

use prelude::*;
use {ControlModel, State};

pub struct DirectVelocity;

impl ControlModel for DirectVelocity {
    type NS = U3;
    type NI = U2;
    type NP = U0;

    fn new() -> Self
    where
        Self: Sized,
    {
        DirectVelocity
    }

    fn state_equation(&self, x: &Vector<U3>, u: &Vector<U2>, _p: &Vector<U0>) -> Vector<U3> {
        let phi = x[2];
        let v = u[0];
        let delta = u[1];

        let (sin_phi, cos_phi) = phi.sin_cos();

        let x_dot = v * cos_phi;
        let y_dot = v * sin_phi;
        let phi_dot = delta;

        Vector3::new(x_dot, y_dot, phi_dot)
    }

    fn linearise(
        &self,
        x0: &Vector<U3>,
        u0: &Vector<U2>,
        _p0: &Vector<U0>,
    ) -> (Matrix<U3, U3>, Matrix<U3, U2>) {
        let v = u0[0];
        let phi = x0[2];

        let (sin_phi, cos_phi) = phi.sin_cos();

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix3::new(
            0.0, 0.0, -v * sin_phi,
            0.0, 0.0, v * cos_phi,
            0.0, 0.0, 0.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B = Matrix3x2::new(
            cos_phi, 0.0,
            sin_phi, 0.0,
            0.0, 1.0,
        );

        (A, B)
    }

    fn linearise_sparsity(&self) -> (Matrix3<bool>, Matrix3x2<bool>) {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A_mask = Matrix3::new(
            false, false, true,
            false, false, true,
            false, false, false,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B_mask = Matrix3x2::new(
            true, false,
            true, false,
            false, true,
        );

        (A_mask, B_mask)
    }

    fn linearise_parameters(
        &self,
        _x0: &Vector<U3>,
        _u0: &Vector<U2>,
        _p0: &Vector<U0>,
    ) -> Matrix<U3, U0> {
        nalgebra::zero()
    }

    fn x_from_state(&self, state: &State) -> Vector<U3> {
        let (x, y) = state.position;
        let phi = state.heading;

        Vector3::new(x, y, phi)
    }

    fn x_to_state(&self, x: &Vector<U3>) -> State {
        State {
            position: (x[0], x[1]),
            heading: x[2],
            velocity: (0., 0.),
        }
    }

    fn input_bounds(&self) -> (Vector<U2>, Vector<U2>) {
        let min = Vector2::new(0.0, -2.0);
        let max = Vector2::new(10.0, 2.0);
        (min, max)
    }

    fn input_delta_bounds(&self) -> (Vector<U2>, Vector<U2>) {
        let min = Vector2::new(NEG_INFINITY, NEG_INFINITY);
        let max = Vector2::new(INFINITY, INFINITY);
        (min, max)
    }
}
