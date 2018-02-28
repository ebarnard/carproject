use nalgebra::{Matrix5, Matrix5x2, MatrixMN, Vector2, Vector5};
use nalgebra::dimension::{U2, U5, U7};

use prelude::*;
use {ControlModel, State};

pub struct NoSlipPoint;

impl ControlModel for NoSlipPoint {
    type NS = U5;
    type NI = U2;
    type NP = U7;

    fn new() -> Self
    where
        Self: Sized,
    {
        NoSlipPoint
    }

    fn name() -> &'static str {
        "no_slip_point"
    }

    fn state_equation(&self, x: &Vector<U5>, u: &Vector<U2>, p: &Vector<U7>) -> Vector<U5> {
        let [phi, v, delta, throttle, delta_target, C1, C2, C3, Cm1, Cm2, Cr1, Cr2] =
            unpack(x, u, p);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let x_dot = v * cos_k;
        let y_dot = v * sin_k;
        let phi_dot = C2 * delta * v;

        let v_dot_motor = throttle * (Cm1 - Cm2 * v);
        let v_dot_friction = -Cr2 * v * v - Cr1 * v.signum();
        let v_dot_cornering = -v * v * delta * delta * C1 * C2;
        let v_dot = v_dot_motor + v_dot_friction + v_dot_cornering;

        let delta_dot = C3 * (delta_target - delta);

        Vector5::new(x_dot, y_dot, phi_dot, v_dot, delta_dot)
    }

    fn linearise(
        &self,
        x0: &Vector<U5>,
        u0: &Vector<U2>,
        p0: &Vector<U7>,
    ) -> (Matrix<U5, U5>, Matrix<U5, U2>) {
        let [phi, v, delta, throttle, _delta_target, C1, C2, C3, Cm1, Cm2, _Cr1, Cr2] =
            unpack(x0, u0, p0);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let dv_dot_dv = -Cm2 * throttle - 2.0 * (C1 * C2 * delta * delta) * v - 2.0 * Cr2 * v;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix5::new(
            0.0, 0.0, -v * sin_k, cos_k, -C1 * v * sin_k,
            0.0, 0.0, v * cos_k, sin_k, C1 * v * cos_k,
            0.0, 0.0, 0.0, C2 * delta, C2 * v,
            0.0, 0.0, 0.0, dv_dot_dv, -2.0 * delta * C1 * C2 * v * v,
            0.0, 0.0, 0.0, 0.0, -C3,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B = Matrix5x2::new(
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            Cm1 - Cm2 * v, 0.0,
            0.0, C3,
        );

        (A, B)
    }

    fn linearise_sparsity(&self) -> (Matrix5<bool>, Matrix5x2<bool>) {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A_mask = Matrix5::new(
            false, false, true, true, true,
            false, false, true, true, true,
            false, false, false, true, true,
            false, false, false, true, true,
            false, false, false, false, true,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B_mask = Matrix5x2::new(
            false, false,
            false, false,
            false, false,
            true, false,
            false, true,
        );

        (A_mask, B_mask)
    }

    fn linearise_parameters(
        &self,
        x0: &Vector<U5>,
        u0: &Vector<U2>,
        p0: &Vector<U7>,
    ) -> Matrix<U5, U7> {
        let [phi, v, delta, throttle, delta_target, C1, C2, _C3, _Cm1, _Cm2, _Cr1, _Cr2] =
            unpack(x0, u0, p0);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let v_delta = v * delta;
        let v_delta_2 = v_delta * v_delta * 0.0;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<U5, U7>::from_row_slice(&[
            -v_delta * sin_k, 0.0, -v * sin_k * C1, 0.0, 0.0, 0.0, 0.0,
            v_delta * cos_k, 0.0, v * cos_k * C1, 0.0, 0.0, 0.0, 0.0,
            0.0, v_delta, C2 * v, 0.0, 0.0, 0.0, 0.0,
            -v_delta_2 * C2, -v_delta_2 * C1, 0.0, throttle, -throttle * v, -v.signum(), -v * v,
            0.0, 0.0, delta_target - delta, 0.0, 0.0, 0.0, 0.0,
        ])
    }

    fn linearise_parameters_sparsity(&self) -> MatrixMN<bool, U5, U7> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        MatrixMN::<bool, U5, U7>::from_row_slice(&[
            true, false, true, false, false, false, false,
            true, false, true, false, true, false, false,
            false, true, true, false, false, false, false,
            true, true, false, true, true, true, true,
            false, false, true, false, false, false, false,
        ])
    }

    fn x_to_state(&self, x: &Vector<U5>) -> State {
        let heading = x[2];
        let v = x[3];

        // TODO: Use slip adjusted heading
        // adj_heading = heading + self.model.params.C1 * control.steering_angle
        // v_x = v * cos(adj_heading)
        // v_y = v * sin(adj_heading)

        let v_x = v * heading.cos();
        let v_y = v * heading.sin();

        State {
            position: (x[0], x[1]),
            heading: heading,
            velocity: (v_x, v_y),
        }
    }

    fn input_bounds(&self) -> (Vector<U2>, Vector<U2>) {
        let min = Vector2::new(0.0, -1.0);
        let max = Vector2::new(1.0, 1.0);
        (min, max)
    }

    fn input_delta_bounds(&self) -> (Vector<U2>, Vector<U2>) {
        let min = Vector2::new(NEG_INFINITY, -0.1);
        let max = Vector2::new(INFINITY, 0.1);
        (min, max)
    }
}

#[inline(always)]
fn unpack(x: &Vector<U5>, u: &Vector<U2>, p: &Vector<U7>) -> [float; 12] {
    // [phi, v, delta, throttle, delta_target, C1, C2, C3, Cm1, Cm2, Cr1, Cr2]
    [
        x[2], x[3], x[4], u[0], u[1], p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    ]
}
