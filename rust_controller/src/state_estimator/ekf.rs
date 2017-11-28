use nalgebra::{self, U3, Vector3};
use nalgebra::linalg::Cholesky;

use prelude::*;
use control_model::{discretise, ControlModel};
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
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP> + Dims2<NM, M::NS>,
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
}

impl<M: ControlModel> StateEstimator<M> for EKF<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP> + Dims2<NM, M::NS>,
{
    fn step(
        &mut self,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> Vector<M::NS> {
        // Use an MLE initial estimate
        if self.initial {
            if let Some(m) = measure {
                let m = Vector3::new(m.position.0, m.position.1, m.heading);
                self.x_hat = nalgebra::zero();
                self.x_hat.fixed_rows_mut::<U3>(0).copy_from(&m);
                self.P = self.Q.clone();

                self.initial = false;
            }

            return self.x_hat.clone();
        }

        // Predict
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

        self.x_hat.clone()
    }
}
