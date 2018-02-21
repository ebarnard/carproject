use nalgebra::{self, DimAdd, DimSum, U0, U3, Vector3};

use prelude::*;
use control_model::{discretise, CombineState, ControlModel};
use {Estimator, Measurement};

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

    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> &Vector<M::NS> {
        if dt == 0.0 {
            return &self.x_hat;
        }

        // Use an MLE initial estimate
        if self.initial {
            if let Some(m) = measure {
                let m = Vector3::new(m.position.0, m.position.1, m.heading);
                self.x_hat = nalgebra::zero();
                self.x_hat.fixed_rows_mut::<U3>(0).copy_from(&m);
                self.P = self.Q.clone();

                self.initial = false;
            }

            return &self.x_hat;
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
            let S_inv = S.try_inverse().expect("S must be invertible");

            // Kalman gain
            let K = &P_predict * H.transpose() * S_inv;

            let x_update = x_predict + &K * y;
            let I = Matrix::<M::NS, M::NS>::identity();
            let IKH = I - &K * H;
            // Use the numerically stable Joseph form to preserve positive-semi-definiteness of P
            let P_update = &IKH * P_predict * IKH.transpose() + &K * self.R * K.transpose();

            self.x_hat = x_update;
            self.P = P_update;
        } else {
            self.x_hat = x_predict;
            self.P = P_predict;
        }

        &self.x_hat
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
    initial_params: Vector<M::NP>,
    Q_initial_params: Matrix<M::NP, M::NP>,
    x_hat: Vector<M::NS>,
    p_hat: Vector<M::NP>,
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
        initial_params: Vector<M::NP>,
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
            initial_params,
            Q_initial_params,
            x_hat: nalgebra::zero(),
            p_hat: nalgebra::zero(),
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
    ) -> (&Vector<M::NS>, &Vector<M::NP>) {
        // TODO: Handle kalman state initialisation better
        let x_combined = if self.inner.initial {
            if let Some(m) = measure {
                let m = Vector3::new(m.position.0, m.position.1, m.heading);
                self.inner.x_hat = nalgebra::zero();
                self.inner.x_hat.fixed_rows_mut::<U3>(0).copy_from(&m);
                self.inner
                    .x_hat
                    .fixed_rows_mut::<M::NP>(M::NS::dim())
                    .copy_from(&self.initial_params);
                self.inner.P = self.inner.Q.clone();
                self.inner
                    .P
                    .fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
                    .copy_from(&self.Q_initial_params);

                self.inner.initial = false;
            }
            &self.inner.x_hat
        } else {
            self.inner.step(
                CombineState::from_ref(model),
                dt,
                u,
                measure,
                &nalgebra::zero(),
            )
        };

        let (x_hat, p_hat) = CombineState::<M>::split_x(x_combined);
        self.x_hat = x_hat;
        self.p_hat = p_hat;
        (&self.x_hat, &self.p_hat)
    }

    fn param_covariance(&self) -> Matrix<M::NP, M::NP> {
        self.inner
            .P
            .fixed_slice::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .into_owned()
    }
}
