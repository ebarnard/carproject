// Kinematic Bicycle Model from Rajesh Rajamani. Vehicle Dynamics and Control.

use nalgebra::dimension::{U2, U4, U7};
use nalgebra::{Matrix4, Matrix4x2, MatrixMN, Vector2, Vector4};

use prelude::*;
use {ControlModel, State};

pub struct KinematicBicycle;

impl ControlModel for KinematicBicycle {
    type NS = U4;
    type NI = U2;
    type NP = U7;

    fn new() -> Self
    where
        Self: Sized,
    {
        KinematicBicycle
    }

    fn name() -> &'static str {
        "kinematic_bicycle"
    }

    fn state_equation(&self, x: &Vector<U4>, u: &Vector<U2>, p: &Vector<U7>) -> Vector<U4> {
        let [phi, v, throttle, wheel, lr_lflr, inv_lr, Cw, Cm1, Cm2, Cr1, Cr2] = unpack(x, u, p);

        let delta = wheel * Cw;
        let beta = (lr_lflr * delta.tan()).atan();
        let (sin_phi_beta, cos_phi_beta) = (phi + beta).sin_cos();
        let sin_beta = beta.sin();

        let x_dot = v * cos_phi_beta;
        let y_dot = v * sin_phi_beta;
        let phi_dot = v * inv_lr * sin_beta;
        let v_dot = throttle * (Cm1 - Cm2 * v) - Cr2 * v * v - Cr1 * v.signum();

        Vector4::new(x_dot, y_dot, phi_dot, v_dot)
    }

    fn linearise(
        &self,
        x0: &Vector<U4>,
        u0: &Vector<U2>,
        p0: &Vector<U7>,
    ) -> (Matrix<U4, U4>, Matrix<U4, U2>) {
        let [phi, v, throttle, wheel, lr_lflr, inv_lr, Cw, Cm1, Cm2, _Cr1, Cr2] =
            unpack(x0, u0, p0);

        let delta = wheel * Cw;
        let (sin_delta, cos_delta) = delta.sin_cos();
        let tan_delta = sin_delta / cos_delta;

        let beta = (lr_lflr * tan_delta).atan();
        let (sin_beta, cos_beta) = beta.sin_cos();

        let (sin_phi_beta, cos_phi_beta) = (phi + beta).sin_cos();

        // tan(beta) = lr_lflr * tan(delta)
        let d_beta_delta = lr_lflr * cos_beta * cos_beta * Cw / (cos_delta * cos_delta);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix4::new(
            0.0, 0.0, -v * sin_phi_beta, cos_phi_beta,
            0.0, 0.0, v * cos_phi_beta, sin_phi_beta,
            0.0, 0.0, 0.0, inv_lr * sin_beta,
            0.0, 0.0, 0.0, -throttle * Cm2 - 2.0 * Cr2 * v,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let B = Matrix4x2::new(
            0.0, -v * sin_phi_beta * d_beta_delta,
            0.0, v * cos_phi_beta * d_beta_delta,
            0.0, v * inv_lr * cos_beta * d_beta_delta,
            Cm1 - Cm2 * v, 0.0,
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
            true, false,
        );

        (A_mask, B_mask)
    }

    fn linearise_parameters(
        &self,
        x0: &Vector<U4>,
        u0: &Vector<U2>,
        p0: &Vector<U7>,
    ) -> Matrix<U4, U7> {
        let [phi, v, throttle, wheel, lr_lflr, inv_lr, Cw, _Cm1, _Cm2, _Cr1, _Cr2] =
            unpack(x0, u0, p0);

        let delta = wheel * Cw;
        let cos_delta = delta.cos();
        let tan_delta = delta.tan();

        let beta = (lr_lflr * tan_delta).atan();
        let (sin_beta, cos_beta) = beta.sin_cos();

        let (sin_phi_beta, cos_phi_beta) = (phi + beta).sin_cos();

        // tan(beta) = lr_lflr * tan(delta)
        let d_beta_lr_lflr = tan_delta * cos_beta * cos_beta;

        // tan(beta) = lr_lflr * tan(Cw * wheel)
        let d_beta_Cw = lr_lflr * cos_beta * cos_beta * wheel / (cos_delta * cos_delta);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<U4, U7>::from_row_slice(&[
            -v * sin_phi_beta * d_beta_lr_lflr, 0.0, -v * sin_phi_beta * d_beta_Cw, 0.0, 0.0, 0.0, 0.0,
            v * cos_phi_beta * d_beta_lr_lflr, 0.0, v * cos_phi_beta * d_beta_Cw, 0.0, 0.0, 0.0, 0.0,
            v * inv_lr * cos_beta * d_beta_lr_lflr, v * sin_beta, v * inv_lr * cos_beta * d_beta_Cw, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, throttle, -throttle * v, -v.signum(), -v * v,
        ])
    }

    fn linearise_parameters_sparsity(&self) -> MatrixMN<bool, U4, U7> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        MatrixMN::<bool, U4, U7>::from_row_slice(&[
            true, false, false, false, false, false, false,
            true, false, false, false, false, false, false,
            true, true, false, false, false, false, false,
            false, false, false, true, true, true, true,
        ])
    }

    fn x_to_state(&self, x: &Vector<U4>) -> State {
        let heading = x[2];
        let v = x[3];

        // TODO: Use actual velocity vector which requires input delta
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
fn unpack(x: &Vector<U4>, u: &Vector<U2>, p: &Vector<U7>) -> [float; 11] {
    // [phi, v, throttle, delta, lr_lflr, inv_lr, w, Cm1, Cm2, Cr1, Cr2]
    [
        x[2], x[3], u[0], u[1], p[0], p[1], p[2], p[3], p[4], p[5], p[6],
    ]
}
