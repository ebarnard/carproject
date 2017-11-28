use nalgebra::{self, U1, U3, Vector3};
use nalgebra::allocator::Reallocator;
use nalgebra::linalg::Cholesky;

use prelude::*;
use control_model::{discretise, ControlModel};
use controller::{Control, State};
use state_estimator::{Measurement, StateEstimator};

type NM = U3;

pub struct EKF<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    // State transition covariance
    Q: Matrix<M::NS, M::NS>,
    // Measurement covariance
    R: Matrix<NM, NM>,
    // Predicted state
    x_hat: Vector<M::NS>,
    // Predicted state covariance
    P: Matrix<M::NS, M::NS>,
    initial: bool,
}

impl<M: ControlModel> EKF<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>
        + Dims2<NM, M::NS>
        + Reallocator<float, U3, U1, M::NS, U1>,
{
    pub fn new(Q: Matrix<M::NS, M::NS>, R: Matrix<NM, NM>) -> EKF<M> {
        EKF {
            Q,
            R,
            x_hat: nalgebra::zero(),
            P: nalgebra::zero(),
            initial: true,
        }
    }

    pub fn predict_update(
        &mut self,
        dt: float,
        control: &Control,
        measure: Option<Measurement>,
        params: &[float],
    ) -> &Vector<M::NS> {
        // Use an MLE initial estimate
        if self.initial {
            if let Some(m) = measure {
                self.x_hat = Vector3::new(m.position.0, m.position.1, m.heading).fixed_resize(0.0);
                self.P = self.Q.clone();

                self.initial = false;
            }

            return &self.x_hat;
        }

        // Predict
        let u = M::u_from_control(control);
        // TODO: Check length of params
        let p = Vector::<M::NP>::from_column_slice(params);
        let (F_c, B_c) = M::linearise(&self.x_hat, &u, &p);
        let (F, _) = discretise(dt, &F_c, &B_c);

        let x_predict = M::step(dt, &self.x_hat, &u, &p);
        let P_predict = &F * &self.P * F.transpose() + &self.Q;

        // Update
        if let Some(m) = measure {
            let H = Matrix::<NM, M::NS>::identity();
            // Ensure there are no angle discontinuities
            let heading = phase_unwrap(x_predict[2], m.heading);
            let z = Vector3::new(m.position.0, m.position.1, heading);

            // Innovation
            let y = z - &H * &x_predict;

            // Innovation covariance
            let S = &H * &P_predict * H.transpose() + &self.R;
            let S_inv = Cholesky::new(S).expect("S must be symmetric positive-definite").inverse();

            // Kalman gain
            let K = &P_predict * H.transpose() * S_inv;

            let x_update = x_predict + &K * y;
            let I = Matrix::<M::NS, M::NS>::identity();
            let P_update = (I - &K * H) * P_predict;

            self.x_hat = x_update;
            self.P = P_update;
        } else {
            self.x_hat = x_predict;
            self.P = P_predict;
        }

        &self.x_hat
    }
}

impl<M: ControlModel> StateEstimator for EKF<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>
        + Dims2<NM, M::NS>
        + Reallocator<float, U3, U1, M::NS, U1>,
{
    fn step(
        &mut self,
        dt: float,
        control: &Control,
        measure: Option<Measurement>,
        params: &[float],
    ) -> State {
        let state = self.predict_update(dt, control, measure, params);
        State {
            position: (state[0], state[1]),
            heading: state[2],
            // TODO: sin and cos or just have a single velocity vector
            velocity: (state[3], 0.0),
        }
    }
}
