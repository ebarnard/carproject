use flame;
use nalgebra::{MatrixMN, U1};
use std::time::Duration;

use prelude::*;
use control_model::{discretise, discretise_sparsity, ControlModel};
use {MpcStage, OsqpMpc};

pub struct MpcBase<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    horizon: u32,
    mpc: OsqpMpc<M::NS, M::NI>,
    model_u_min: Vector<M::NI>,
    model_u_max: Vector<M::NI>,
    base_stage: MpcBaseStage,
}

pub struct MpcBaseStage {
    pub x_target: Vector<Dy>,
    pub x_linear_penalty: Vector<Dy>,
    pub stage_ineq: Matrix<Dy, Dy>,
    pub stage_ineq_min: Vector<Dy>,
    pub stage_ineq_max: Vector<Dy>,
}

impl MpcBaseStage {
    fn zero(&mut self) {
        self.x_target.fill(0.0);
        self.x_linear_penalty.fill(0.0);
        self.stage_ineq.fill(0.0);
        self.stage_ineq_min.fill(0.0);
        self.stage_ineq_max.fill(0.0);
    }
}

impl<M: ControlModel> MpcBase<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(
        model: &M,
        N: u32,
        Q: Matrix<Dy, Dy>,
        Q_terminal: Matrix<Dy, Dy>,
        R: Matrix<M::NI, M::NI>,
        stage_ineq_sparsity: &MatrixMN<bool, Dy, Dy>,
    ) -> MpcBase<M> {
        // Some components of A and B will always be zero and can be excluded from the sparse
        // constraint matrix
        let (A_sparsity, B_sparsity) = model.linearise_sparsity();
        let (A_d_sparsity, B_d_sparsity) = discretise_sparsity(&A_sparsity, &B_sparsity);

        let guard = flame::start_guard("osqp mpc create");

        let mut mpc = OsqpMpc::new(
            N as usize,
            Q,
            Q_terminal,
            R,
            &A_d_sparsity,
            &B_d_sparsity,
            stage_ineq_sparsity,
        );
        let (input_min, input_max) = model.input_bounds();
        mpc.set_input_bounds(input_min.clone(), input_max.clone());
        let (input_delta_min, input_delta_max) = model.input_delta_bounds();
        mpc.set_input_delta_bounds(input_delta_min, input_delta_max);

        let (n_stage_ineq, ns_plus_nvs) = stage_ineq_sparsity.shape();
        let base_stage = MpcBaseStage {
            x_target: Vector::zeros_generic(Dy::new(ns_plus_nvs), U1),
            x_linear_penalty: Vector::zeros_generic(Dy::new(ns_plus_nvs), U1),
            stage_ineq: Matrix::zeros_generic(Dy::new(n_stage_ineq), Dy::new(ns_plus_nvs)),
            stage_ineq_min: Vector::zeros_generic(Dy::new(n_stage_ineq), U1),
            stage_ineq_max: Vector::zeros_generic(Dy::new(n_stage_ineq), U1),
        };

        guard.end();

        MpcBase {
            horizon: N,
            mpc,
            model_u_min: input_min,
            model_u_max: input_max,
            base_stage,
        }
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
        time_limit: Duration,
        mut f: F,
    ) -> (&Matrix<M::NI, Dy>, &Matrix<M::NS, Dy>)
    where
        F: FnMut(usize, &Vector<M::NS>, &Vector<M::NI>, &mut MpcBaseStage),
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
            let base_stage = &mut self.base_stage;
            base_stage.zero();
            f(i, &x_i, &u_i, base_stage);

            let stage = MpcStage {
                A: &A,
                B: &B,
                x0: &x_i,
                u0: &u_i,
                x_target: &base_stage.x_target,
                x_linear_penalty: &base_stage.x_linear_penalty,
                stage_ineq: &base_stage.stage_ineq,
                stage_ineq_min: &base_stage.stage_ineq_min,
                stage_ineq_max: &base_stage.stage_ineq_max,
            };

            // Give the values to the builder
            let guard = flame::start_guard("update mpc matrices");
            self.mpc.set_model(i, &stage);
            guard.end();
        }
        guard.end();

        let guard = flame::start_guard("mpc solve");
        let solution = self.mpc
            .solve(u.column(0).into_owned(), time_limit)
            .expect("solve failed");
        guard.end();

        (&solution.u, &solution.x)
    }
}
