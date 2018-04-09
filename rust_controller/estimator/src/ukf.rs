use nalgebra::storage::Storage;
use nalgebra::{self, DimNameSum, SymmetricEigen, U1, Vector3};

use control_model::ControlModel;
use prelude::*;
use {Estimator, Measurement, NM};

pub struct JointUkf<M: ControlModel>
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

impl<M: ControlModel> Estimator<M> for JointUkf<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn new(
        initial_params: Vector<M::NP>,
        Q_state: Matrix<M::NS, M::NS>,
        Q_params: Matrix<M::NP, M::NP>,
        Q_initial_params: Matrix<M::NP, M::NP>,
        R: Matrix<NM, NM>,
    ) -> JointUkf<M> {
        let mut Q = Matrix::<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>::zeros();
        Q.fixed_slice_mut::<M::NS, M::NS>(0, 0).copy_from(&Q_state);
        Q.fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .copy_from(&Q_params);

        let mut P = Q.clone();
        P.fixed_slice_mut::<M::NP, M::NP>(M::NS::dim(), M::NS::dim())
            .copy_from(&Q_initial_params);

        // Ensure P is positive-definite
        let P = P + Matrix::from_diagonal(&Vector::<DimNameSum<M::NP, M::NS>>::from_element(1e-9));

        JointUkf {
            Q,
            R,
            x_hat: nalgebra::zero(),
            p_hat: initial_params,
            P,
            initial: true,
        }
    }

    fn name() -> &'static str {
        "joint_ukf"
    }

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
            let L_ = M::NS::dim() + M::NP::dim();
            let L = L_ as float;
            let alpha = 1e-3;
            let beta = 2.0;
            let kappa = 0.0;
            let lambda = alpha * alpha * (L + kappa) - L;

            let (P_nrows, P_ncols) = self.P.data.shape();
            let P_eigen = SymmetricEigen::new(to_dynamic(&self.P));
            let P_eigenvectors = from_dynamic(P_nrows, P_ncols, &P_eigen.eigenvectors);
            let sqrt_P_diag = from_dynamic_vec(P_nrows, &P_eigen.eigenvalues).map(|v| v.sqrt());
            let sqrt_P =
                &P_eigenvectors * Matrix::from_diagonal(&sqrt_P_diag) * P_eigenvectors.transpose();

            let w_m_0 = lambda / (L + lambda);
            let w_c_0 = w_m_0 + (1.0 - alpha * alpha + beta);
            let w = 1.0 / (2.0 * (L + lambda));

            let mut y_0 = Vector::<DimNameSum<M::NP, M::NS>>::zeros();
            y_0.fixed_rows_mut::<M::NS>(0)
                .copy_from(&model.step(dt, &self.x_hat, u, &self.p_hat));
            y_0.fixed_rows_mut::<M::NP>(M::NS::dim())
                .copy_from(&self.p_hat);

            let scaling = (L + lambda).sqrt();
            let eval_y = |i, scaling| {
                let x_i = &self.x_hat + scaling * sqrt_P.fixed_slice::<U1, M::NS>(i, 0).transpose();
                let p_i = &self.p_hat
                    + scaling * sqrt_P.fixed_slice::<U1, M::NP>(i, M::NS::dim()).transpose();
                let mut y_i = Vector::<DimNameSum<M::NP, M::NS>>::zeros();
                y_i.fixed_rows_mut::<M::NS>(0)
                    .copy_from(&model.step(dt, &x_i, u, &p_i));
                y_i.fixed_rows_mut::<M::NP>(M::NS::dim()).copy_from(&p_i);
                y_i
            };

            let mut y_plus = Matrix::<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>::zeros();
            let mut y_minus = Matrix::<DimNameSum<M::NP, M::NS>, DimNameSum<M::NP, M::NS>>::zeros();
            for i in 0..L_ {
                y_plus.column_mut(i).copy_from(&eval_y(i, scaling));
                y_minus
                    .column_mut(i)
                    .copy_from(&eval_y(L_ - i - 1, -scaling));
            }

            let mut y_bar = w_m_0 * &y_0;
            for i in 0..L_ {
                y_bar += w * (y_plus.column(i) + y_minus.column(i));
            }

            let mean_offset = y_0 - &y_bar;
            let mut P_predict = w_c_0 * &mean_offset * mean_offset.transpose();
            for i in 0..L_ {
                let mean_offset = y_plus.column(i) - &y_bar;
                P_predict += w * &mean_offset * mean_offset.transpose();
                let mean_offset = y_minus.column(i) - &y_bar;
                P_predict += w * &mean_offset * mean_offset.transpose();
            }
            P_predict += &self.Q;

            let x_predict = y_bar.fixed_rows::<M::NS>(0).into_owned();
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
