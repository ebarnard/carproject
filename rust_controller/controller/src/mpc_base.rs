use flame;
use nalgebra::VectorN;

use prelude::*;
use control_model::{discretise, discretise_sparsity, ControlModel};
use OsqpMpc;

pub struct MpcBase<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    horizon: u32,
    mpc: OsqpMpc<M::NS, M::NI>,
    model_u_min: Vector<M::NI>,
    model_u_max: Vector<M::NI>,
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

        let guard = flame::start_guard("osqp mpc create");

        let mut mpc = OsqpMpc::new(
            N as usize,
            Q,
            R,
            &A_d_sparsity,
            &B_d_sparsity,
            stage_ineq_sparsity,
        );
        let (input_min, input_max) = model.input_bounds();
        mpc.set_input_bounds(input_min.clone(), input_max.clone());
        let (input_delta_min, input_delta_max) = model.input_delta_bounds();
        mpc.set_input_delta_bounds(input_delta_min, input_delta_max);

        guard.end();

        MpcBase {
            horizon: N,
            mpc,
            model_u_min: input_min,
            model_u_max: input_max,
        }
    }

    pub fn horizon_len(&self) -> u32 {
        self.horizon
    }

    pub fn update_input_bounds(&mut self, ext_u_min: Vector<M::NI>, ext_u_max: Vector<M::NI>) {
        let u_min = self.model_u_min.zip_map(&ext_u_min, max);
        let u_max = self.model_u_max.zip_map(&ext_u_max, min);
        self.mpc.set_input_bounds(u_min, u_max);
    }

    pub fn step<F>(
        &mut self,
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        u: &Matrix<M::NI, Dy>,
        p: &Vector<M::NP>,
        mut f: F,
    ) -> (&Matrix<M::NI, Dy>, &Matrix<M::NS, Dy>)
    where
        F: FnMut(usize, &Vector<M::NS>, &Vector<M::NI>, &mut OsqpMpc<M::NS, M::NI>)
            -> (Vector<M::NS>, Vector<M::NS>),
    {
        let _guard = flame::start_guard("controller step");
        let N = self.horizon as usize;
        let mut x_i = x.clone();

        let guard = flame::start_guard("mpc setup");
        for i in 0..N {
            let u_i = u.column(i + 1).into_owned();

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

        let guard = flame::start_guard("mpc solve");
        let solution = self.mpc
            .solve(u.column(0).into_owned())
            .expect("solve failed");
        guard.end();

        (&solution.u, &solution.x)
    }
}
