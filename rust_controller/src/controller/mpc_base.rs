use flame;
use nalgebra::{Dynamic, MatrixMN, VectorN};

use prelude::*;
use controller::OsqpMpc;
use control_model::{discretise, discretise_sparsity, ControlModel};

pub struct MpcBase<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    horizon: u32,
    u_mpc: Matrix<M::NI, Dynamic>,
    mpc: OsqpMpc<M::NS, M::NI>,
}

impl<M: ControlModel> MpcBase<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(
        model: &M,
        N: u32,
        Q: Matrix<M::NS, M::NS>,
        R: Matrix<M::NI, M::NI>,
        stage_ineq_sparsity: &[VectorN<bool, M::NS>],
    ) -> MpcBase<M> {
        // Some components of A and B will always be zero and can be excluded from the sparse
        // constraint matrix
        let (A_sparsity, B_sparsity) = model.linearise_sparsity();
        let (A_d_sparsity, B_d_sparsity) = discretise_sparsity(&A_sparsity, &B_sparsity);

        let mpc = flame::span_of("osqp mpc create", || {
            let mut mpc = OsqpMpc::new(
                N as usize,
                Q,
                R,
                &A_d_sparsity,
                &B_d_sparsity,
                stage_ineq_sparsity,
            );
            let (input_min, input_max) = model.input_bounds();
            mpc.set_input_bounds(input_min, input_max);
            let (input_delta_min, input_delta_max) = model.input_delta_bounds();
            mpc.set_input_delta_bounds(input_delta_min, input_delta_max);
            mpc
        });

        MpcBase {
            horizon: N,
            u_mpc: MatrixMN::zeros_generic(<M::NI as DimName>::name(), Dynamic::new(N as usize)),
            mpc,
        }
    }

    pub fn step<F>(
        &mut self,
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        p: &Vector<M::NP>,
        mut f: F,
    ) -> (Vector<M::NI>, Vector<M::NS>)
    where
        F: FnMut(usize, &Vector<M::NS>, &Vector<M::NI>, &mut OsqpMpc<M::NS, M::NI>)
            -> (Vector<M::NS>, Vector<M::NS>),
    {
        let _guard = flame::start_guard("controller step");
        let N = self.horizon as usize;
        let mut x_i = x.clone();

        let guard = flame::start_guard("mpc setup");
        for i in 0..N {
            let u_i = self.u_mpc.column(i).into_owned();

            // Linearise model around x_i-1 and u_i
            let (A_c, B_c) = flame::span_of("model linearise", || model.linearise(&x_i, &u_i, p));
            let (A, B) = flame::span_of("model discretise", || discretise(dt, &A_c, &B_c));

            // Integrate state using nonlinear model
            x_i = flame::span_of("model integrate", || model.step(dt, &x_i, &u_i, p));

            // Get state target, state linear penalty and set any inequalities
            let (x_target, x_linear_penalty) = f(i, &x_i, &u_i, &mut self.mpc);

            // Give the values to the builder
            flame::span_of("update mpc matrices", || {
                self.mpc
                    .set_model(i, &A, &B, &x_i, &u_i, &x_target, &x_linear_penalty)
            });
        }
        guard.end();

        let mpc = &mut self.mpc;
        let solution = flame::span_of("mpc solve", || mpc.solve()).expect("solve failed");

        self.u_mpc
            .columns_mut(0, N - 1)
            .copy_from(&solution.u.columns(1, N - 1));
        self.u_mpc
            .column_mut(N - 1)
            .copy_from(&solution.u.column(N - 1));

        (
            solution.u.column(0).into_owned(),
            solution.x.column(0).into_owned(),
        )
    }
}
