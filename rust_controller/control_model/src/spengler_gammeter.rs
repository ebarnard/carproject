use nalgebra::{Matrix4, Matrix4x2, Matrix4x6, Vector2, Vector4};
use nalgebra::dimension::{U2, U4, U6};

use prelude::*;
use {ControlModel, State};

pub struct SpenglerGammeter;

impl ControlModel for SpenglerGammeter {
    type NS = U4;
    type NI = U2;
    type NP = U6;

    fn new() -> Self
    where
        Self: Sized,
    {
        SpenglerGammeter
    }

    fn name() -> &'static str {
        "spengler_gammeter"
    }

    fn state_equation(&self, x: &Vector<U4>, u: &Vector<U2>, p: &Vector<U6>) -> Vector<U4> {
        let [phi, v, throttle, delta, C1, C2, Cm1, Cm2, Cr1, Cr2] = unpack(x, u, p);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let x_dot = v * cos_k;
        let y_dot = v * sin_k;
        let phi_dot = C2 * delta * v;

        let v_dot_motor = Cm1 * throttle - Cm2 * throttle * v;
        let v_dot_friction = -Cr2 * v * v - Cr1 * v.signum();
        let v_dot_cornering = -v * v * delta * delta * C1 * C2;
        let v_dot = v_dot_motor + v_dot_friction + v_dot_cornering;

        Vector4::new(x_dot, y_dot, phi_dot, v_dot)
    }

    fn linearise(
        &self,
        x0: &Vector<U4>,
        u0: &Vector<U2>,
        p0: &Vector<U6>,
    ) -> (Matrix<U4, U4>, Matrix<U4, U2>) {
        let [phi, v, throttle, delta, C1, C2, Cm1, Cm2, _Cr1, Cr2] = unpack(x0, u0, p0);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix4::new(
            0.0, 0.0, -v * sin_k, cos_k,
            0.0, 0.0, v * cos_k, sin_k,
            0.0, 0.0, 0.0, C2 * delta,
            0.0, 0.0, 0.0, -Cm2 * throttle - 2.0 * (Cr2 + C1 * C2 * delta * delta) * v
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B = Matrix4x2::new(
            0.0, -C1 * v * sin_k,
            0.0, C1 * v * cos_k,
            0.0, C2 * v,
            Cm1 - Cm2 * v, -2.0 * C1 * C2 * v * v * delta
        );

        (A, B)
    }

    fn linearise_sparsity(&self) -> (Matrix4<bool>, Matrix4x2<bool>) {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A_mask = Matrix4::new(
            false, false, true, true,
            false, false, true, true,
            false, false, false, true,
            false, false, false, true,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B_mask = Matrix4x2::new(
            false, true,
            false, true,
            false, true,
            true, true,
        );

        (A_mask, B_mask)
    }

    fn linearise_parameters(
        &self,
        x0: &Vector<U4>,
        u0: &Vector<U2>,
        p0: &Vector<U6>,
    ) -> Matrix<U4, U6> {
        let [phi, v, throttle, delta, C1, C2, _Cm1, _Cm2, _Cr1, _Cr2] = unpack(x0, u0, p0);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let v_delta = v * delta;
        let v_delta_2 = v_delta * v_delta;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4x6::new(
            -v_delta * sin_k, 0.0, 0.0, 0.0, 0.0, 0.0,
            v_delta * cos_k, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, v_delta, 0.0, 0.0, 0.0, 0.0,
            -v_delta_2 * C2, -v_delta_2 * C1, throttle, -throttle * v, -v.signum(), -v * v
        )
    }

    fn linearise_parameters_sparsity(&self) -> Matrix4x6<bool> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4x6::new(
            true, false, false, false, false, false,
            true, false, false, true, false, false,
            false, true, false, false, false, false,
            true, true, true, true, false, true,
        )
    }

    fn x_to_state(&self, x: &Vector<U4>) -> State {
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
fn unpack(x: &Vector<U4>, u: &Vector<U2>, p: &Vector<U6>) -> [float; 10] {
    // [phi, v, throttle, delta, C1, C2, Cm1, Cm2, Cr1, Cr2]
    [x[2], x[3], u[0], u[1], p[0], p[1], p[2], p[3], p[4], p[5]]
}
