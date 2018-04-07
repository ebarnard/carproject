use nalgebra::{self, DimNameSum, Vector3};

use control_model::{discretise, ControlModel};
use prelude::*;
use {Estimator, Measurement, NM};

pub struct JointEKF<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    Q: Matrix<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>,
    R: Matrix<NM, NM>,
    x_hat: Vector<M::NS>,
    p_hat: Vector<M::NP>,
    P: Matrix<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>,
    initial: bool,
}

impl<M: ControlModel> JointEKF<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    pub fn new(
        initial_params: Vector<M::NP>,
        Q_state: Matrix<M::NS, M::NS>,
        Q_params: Matrix<M::NP, M::NP>,
        Q_initial_params: Matrix<M::NP, M::NP>,
        R: Matrix<NM, NM>,
    ) -> JointEKF<M> {
        let mut Q = Matrix::<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>::zeros();
        Q.fixed_slice_mut::<M::NS, M::NS>(0, 0).copy_from(&Q_state);
        Q.fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .copy_from(&Q_params);

        let mut P = Q.clone();
        P.fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .copy_from(&Q_initial_params);

        JointEKF {
            Q,
            R,
            x_hat: nalgebra::zero(),
            p_hat: initial_params,
            P,
            initial: true,
        }
    }
}

impl<M: ControlModel> Estimator<M> for JointEKF<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
    ) -> (
        &Vector<M::NS>,
        &Vector<M::NP>,
        &Matrix<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>,
    ) {
        if self.initial {
            if let Some(m) = measure {
                let m = Vector3::new(m.position.0, m.position.1, m.heading);
                self.x_hat.fixed_rows_mut::<NM>(0).copy_from(&m);
                self.initial = false;
            }
            return (&self.x_hat, &self.p_hat, &self.P);
        }

        // Predict
        let (x_predict, P_predict) = if dt == 0.0 {
            (self.x_hat.clone(), self.P.clone())
        } else {
            // Joint state evolution discretisation matrix looks like:
            // | delta_x_k | = | A  P | | delta_x_k-1 |
            // | delta_p_k | = | 0  I | | delta_p_k-1 |
            let (A_c, _) = model.linearise(&self.x_hat, u, &self.p_hat);
            let P_c = model.linearise_parameters(&self.x_hat, u, &self.p_hat);

            let (F_A, F_P) = discretise(dt, &A_c, &P_c);

            let mut F = Matrix::<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>::identity();
            F.fixed_slice_mut::<M::NS, M::NS>(0, 0).copy_from(&F_A);
            F.fixed_slice_mut::<M::NS, M::NP>(0, M::NS::dim())
                .copy_from(&F_P);

            let x_predict = model.step(dt, &self.x_hat, u, &self.p_hat);
            let P_predict = &F * &self.P * F.transpose() + &self.Q;
            (x_predict, P_predict)
        };

        // Update
        if let Some(m) = measure {
            let H = Matrix::<NM, DimNameSum<M::NP, M::NS>>::identity();
            // Ensure there are no angle discontinuities
            let heading = phase_unwrap(x_predict[2], m.heading);
            let z = Vector3::new(m.position.0, m.position.1, heading);

            // Innovation
            let y = z - H.fixed_columns::<M::NS>(0) * &x_predict;

            // Innovation covariance
            let S = &H * &P_predict * H.transpose() + self.R;
            let S_inv = S.try_inverse().expect("S must be invertible");

            // Kalman gain
            let K = &P_predict * H.transpose() * S_inv;

            let x_update = x_predict + K.fixed_rows::<M::NS>(0) * y;
            let p_update = &self.p_hat + K.fixed_rows::<M::NP>(M::NS::dim()) * y;

            let I = Matrix::<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>::identity();
            let IKH = I - &K * H;
            // Use the numerically stable Joseph form to preserve positive-semi-definiteness of P
            let P_update = &IKH * P_predict * IKH.transpose() + &K * self.R * K.transpose();

            self.x_hat = x_update;
            self.p_hat = p_update;
            self.P = P_update;
        } else {
            self.x_hat = x_predict;
            self.P = P_predict;
        }

        (&self.x_hat, &self.p_hat, &self.P)
    }
}
