use nalgebra::{Matrix4, Matrix4x2, Matrix4x6, Vector2, Vector4};
use nalgebra::dimension::{U2, U4, U6};

use prelude::*;
use controller::State;
use control_model::{ControlModel, SimulationControlModel};

pub struct SpenglerGammeterBicycle;

impl SpenglerGammeterBicycle {
    fn vals(
        &self,
        x: &Vector<U4>,
        u: &Vector<U2>,
        p: &Vector<U6>,
    ) -> (float, float, float, float, float, float, float, float, float) {
        let phi = x[2];
        let v = x[3];

        let throttle = u[0];
        let delta = u[1];

        let C1 = p[0];
        let C2 = p[1];
        let Cm1 = p[2];
        let Cm2 = p[3];
        let Cr2 = p[5];

        (phi, v, throttle, delta, C1, C2, Cm1, Cm2, Cr2)
    }
}

impl ControlModel for SpenglerGammeterBicycle {
    type NS = U4;
    type NI = U2;
    type NP = U6;

    fn state_equation(&self, x: &Vector<U4>, u: &Vector<U2>, p: &Vector<U6>) -> Vector<U4> {
        let (phi, v, throttle, delta, C1, C2, Cm1, Cm2, Cr2) = self.vals(x, u, p);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let x_dot = v * cos_k;
        let y_dot = v * sin_k;
        let phi_dot = C2 * delta * v;

        let v_dot_motor = Cm1 * throttle - Cm2 * throttle * v;
        let v_dot_friction = -Cr2 * v * v;
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
        let (phi, v, throttle, delta, C1, C2, Cm1, Cm2, Cr2) = self.vals(x0, u0, p0);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let A = Matrix4::new(
            0.0, 0.0, -v * sin_k, cos_k,
            0.0, 0.0, v * cos_k, sin_k,
            0.0, 0.0, 0.0, C2 * delta,
            0.0, 0.0, 0.0, -Cm2 * throttle - 2.0 * (Cr2 + C1 * C2 * delta * delta) * v
        );

        let B = Matrix4x2::new(
            0.0, -C1 * v * sin_k,
            0.0, C1 * v * cos_k,
            0.0, C2 * v,
            Cm1 - Cm2 * v, -2.0 * C1 * C2 * v * v * delta
        );

        (A, B)
    }

    fn linearise_nonzero_mask(&self) -> (Matrix4<bool>, Matrix4x2<bool>) {
        let A_mask = Matrix4::new(
            false, false, true, true,
            false, false, true, true,
            false, false, false, true,
            false, false, false, true,
        );

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
        let (phi, v, throttle, delta, C1, C2, _Cm1, _Cm2, _Cr2) = self.vals(x0, u0, p0);

        let (sin_k, cos_k) = (phi + C1 * delta).sin_cos();

        let v_delta = v * delta;
        let v_delta_2 = v_delta * v_delta;

        Matrix4x6::new(
            -v_delta * sin_k, 0.0, 0.0, 0.0, 0.0, 0.0,
            v_delta * cos_k, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, v_delta, 0.0, 0.0, 0.0, 0.0,
            -v_delta_2 * C2, -v_delta_2 * C1, throttle, -throttle * v, 0.0, -v * v
        )
    }

    fn linearise_parameters_sparsity(&self) -> Matrix4x6<bool> {
        Matrix4x6::new(
            true, false, false, false, false, false,
            true, false, false, true, false, false,
            false, true, false, false, false, false,
            true, true, true, true, false, true,
        )
    }

    fn x_from_state(&self, state: &State) -> Vector<U4> {
        let (x, y) = state.position;
        let phi = state.heading;
        let v = float::hypot(state.velocity.0, state.velocity.1);

        Vector4::new(x, y, phi, v)
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
}

impl SimulationControlModel for SpenglerGammeterBicycle {
    fn params(&self) -> &[float] {
        &[
            // C1: Steering slip - negative for oversteer, positive for understeer
            0.1,
            // C2: Steering angle coupling - radians turned / meter travelled
            5.0,
            // Cm1: Duty cycle to acceleration a = Cm1 * throttle
            1.0,
            // Cm2 = Cm1 / v_motor_max (i.e. max speed with no air resistance)
            0.5,
            // Cr1: Rolling resistance
            0.0,
            // Cr2: Reduced air resistance coefficient (0.5 * rho * A * C_d)
            0.0,
        ]
    }
}
