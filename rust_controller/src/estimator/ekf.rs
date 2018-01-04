use nalgebra::{self, DimAdd, DimSum, U0, U3, Vector3};
use nalgebra::linalg::Cholesky;

use prelude::*;
use control_model::{discretise, CombineState, ControlModel};
use estimator::{Estimator, Measurement};

type NM = U3;

pub struct EKF<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP> + Dims2<NM, M::NS>,
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

impl<M: ControlModel> Estimator<M> for EKF<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP> + Dims2<NM, M::NS>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NS>, Vector<M::NP>) {
        // Use an MLE initial estimate
        if self.initial {
            if let Some(m) = measure {
                let m = Vector3::new(m.position.0, m.position.1, m.heading);
                self.x_hat = nalgebra::zero();
                self.x_hat.fixed_rows_mut::<U3>(0).copy_from(&m);
                self.P = self.Q.clone();

                self.initial = false;
            }

            return (self.x_hat.clone(), p.clone());
        }

        // Predict
        let (F_c, B_c) = model.linearise(&self.x_hat, u, p);
        let (F, _) = discretise(dt, &F_c, &B_c);

        let x_predict = model.step(dt, &self.x_hat, u, p);
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
            let S = &H * &P_predict * H.transpose() + self.R;
            let S_inv = Cholesky::new(S)
                .expect("S must be symmetric positive-definite")
                .inverse();

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

        (self.x_hat.clone(), p.clone())
    }

    fn param_covariance(&self) -> Matrix<M::NP, M::NP> {
        unimplemented!();
    }
}

pub struct JointEKF<M: ControlModel>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<NM, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    inner: EKF<CombineState<M>>,
    Q_initial_params: Matrix<M::NP, M::NP>,
}

impl<M: ControlModel> JointEKF<M>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<NM, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    pub fn new(
        Q_state: Matrix<M::NS, M::NS>,
        Q_params: Matrix<M::NP, M::NP>,
        Q_initial_params: Matrix<M::NP, M::NP>,
        R: Matrix<NM, NM>,
    ) -> JointEKF<M> {
        let mut Q: Matrix<DimSum<M::NS, M::NP>, DimSum<M::NS, M::NP>> = nalgebra::zero();
        Q.fixed_slice_mut::<M::NS, M::NS>(0, 0).copy_from(&Q_state);
        Q.fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .copy_from(&Q_params);

        JointEKF {
            inner: EKF::new(Q, R),
            Q_initial_params,
        }
    }
}

impl<M: ControlModel> Estimator<M> for JointEKF<M>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0>
        + Dims3<M::NS, M::NI, M::NP>
        + Dims2<NM, DimSum<M::NS, M::NP>>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NS>, Vector<M::NP>) {
        // TODO: Handle kalman state initialisation better
        if self.inner.initial {
            if let Some(m) = measure {
                let m = Vector3::new(m.position.0, m.position.1, m.heading);
                self.inner.x_hat = nalgebra::zero();
                self.inner.x_hat.fixed_rows_mut::<U3>(0).copy_from(&m);
                self.inner
                    .x_hat
                    .fixed_rows_mut::<M::NP>(M::NS::dim())
                    .copy_from(p);
                self.inner.P = self.inner.Q.clone();
                self.inner
                    .P
                    .fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
                    .copy_from(&self.Q_initial_params);

                self.inner.initial = false;
            }

            return CombineState::<M>::split_x(&self.inner.x_hat);
        }

        let x_combined = self.inner
            .step(
                CombineState::from_ref(model),
                dt,
                u,
                measure,
                &nalgebra::zero(),
            )
            .0;
        CombineState::<M>::split_x(&x_combined)
    }

    fn param_covariance(&self) -> Matrix<M::NP, M::NP> {
        self.inner
            .P
            .fixed_slice::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .into_owned()
    }
}
