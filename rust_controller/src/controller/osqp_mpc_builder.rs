use flame;
use itertools::{repeat_n, Itertools};
use log::LogLevel::Debug;
use nalgebra::{self, Dynamic as Dy, MatrixMN, U1, VectorN};
use osqp::{Problem, Settings, Status};
use std::iter::once;

use prelude::*;
use sparse;

pub struct OsqpMpc<NS: DimName, NI: DimName>
where
    DefaultAllocator: Dims2<NS, NI>,
{
    problem: Problem,
    N: usize,
    n_stage_ineq: usize,
    stage_ineq_lambda_max: float,
    // Objective:
    R: Matrix<Dy, Dy>,
    q: Vector<Dy>,
    Q_stage: Matrix<NS, NS>,
    R_stage_grad: Vector<NI>,
    // Inequalities:
    // N * ns state transition rows
    // N * ni input constraints
    A: sparse::CscMatrix,
    l: Vector<Dy>,
    u: Vector<Dy>,
    A_blocks: Vec<sparse::BlockRef<NS, NS>>,
    B_blocks: Vec<sparse::BlockRef<NS, NI>>,
    stage_ineq_blocks: Vec<sparse::BlockRef<U1, NS>>,
    u_min: Vector<NI>,
    u_max: Vector<NI>,
    u_delta_min: Vector<NI>,
    u_delta_max: Vector<NI>,
    // Optimisation results
    x_mpc: Matrix<NS, Dy>,
    u_mpc: Matrix<NI, Dy>,
    u_prev: Vector<NI>,
}

impl<NS: DimName, NI: DimName> OsqpMpc<NS, NI>
where
    DefaultAllocator: Dims2<NS, NI>,
{
    pub fn new(
        N: usize,
        Q_stage: Matrix<NS, NS>,
        R_stage_grad: Vector<NI>,
        A_sparsity: &MatrixMN<bool, NS, NS>,
        B_sparsity: &MatrixMN<bool, NS, NI>,
        stage_ineq_sparsity: &[VectorN<bool, NS>],
    ) -> OsqpMpc<NS, NI> {
        let N = N as usize;
        let ns = NS::dim();
        let ni = NI::dim();
        let n_stage_ineq = stage_ineq_sparsity.len();

        // Build state quadratic penalty
        let Q = sparse::block(&Q_stage);
        let Q = sparse::block_diag(&repeat_n(&Q, N).collect_vec());

        // Build input delta quadratic penalty
        let R_grad = Matrix::from_diagonal(&R_stage_grad);
        let R_grad = sparse::block(&R_grad);
        let R_diag = sparse::block_diag(&repeat_n(&R_grad, N - 1).collect_vec());
        let R_main_diag = sparse::block_diag(&[&(&R_diag + &R_diag), &R_grad]);
        let R_min_diag = -R_diag;
        let R_sub_diag = sparse::bmat(&[
            &[None, Some(&sparse::zeros(ni, ni))],
            &[Some(&R_min_diag), None],
        ]);
        let R_super_diag = sparse::bmat(&[
            &[None, Some(&R_min_diag)],
            &[Some(&sparse::zeros(ni, ni)), None],
        ]);
        let mut R = R_main_diag + R_sub_diag + R_super_diag;

        // Build penalty matrix P
        let P = sparse::block_diag(&[
            &Q,
            &R,
            // Soft stage inequality penalty variable has no quadratic cost
            &sparse::zeros(N * n_stage_ineq, N * n_stage_ineq),
        ]).build_csc();

        // Build state evolution matrices
        let (Ax, A_blocks): (Vec<_>, Vec<_>) =
            (1..N).map(|_| sparse::block_mut(A_sparsity)).unzip();
        let (Au, B_blocks): (Vec<_>, Vec<_>) =
            (0..N).map(|_| sparse::block_mut(B_sparsity)).unzip();

        let Ax = -sparse::eye(N * ns)
            + sparse::bmat(&[
                &[None, Some(sparse::zeros(ns, ns))],
                &[Some(sparse::block_diag(&Ax)), None],
            ]);
        let Au = sparse::block_diag(&Au);

        // Build stage soft inequality matrix
        // Soft inequalities are formulated as:
        // 0         < b_i           < INFINITY
        // min       < F_i * x + b_i < INFINITY
        // -INFINITY < F_i * x - b_i < max
        let mut stage_ineq_blocks = Vec::with_capacity(N * n_stage_ineq);
        let stage_ineqs = sparse::block_diag(&(0..N)
            .map(|_| {
                sparse::vstack(&stage_ineq_sparsity
                    .iter()
                    .map(|ineq_sparsity| {
                        let ineq_sparsity = ineq_sparsity.transpose();
                        let (ineq, block) = sparse::block_mut(&ineq_sparsity);
                        stage_ineq_blocks.push(block);
                        ineq
                    })
                    // Ensure stage_ineqs has N * ns columns
                    .chain(once(sparse::zeros(0, ns)))
                    .collect::<Vec<_>>())
            })
            .collect::<Vec<_>>());

        let stage_ineq_diag = sparse::block_diag(&(0..N)
            .map(|_| {
                sparse::vstack(&stage_ineq_sparsity
                    .iter()
                    .map(|_| sparse::eye(1))
                    .collect::<Vec<_>>())
            })
            .collect::<Vec<_>>());

        // Build constraint matrix A
        let A = sparse::bmat(&[
            // State evolution
            &[Some(&Ax), Some(&Au), None],
            // Input absolute and delta constraints
            &[None, Some(&sparse::eye(N * ni)), None],
            // Soft stage inequality min constraints
            &[Some(&stage_ineqs), None, Some(&stage_ineq_diag)],
            // Soft stage inequality max constraints
            &[Some(&stage_ineqs), None, Some(&-&stage_ineq_diag)],
            // Soft stage inequality penalty variable is positive constraint
            &[None, None, Some(&sparse::eye(N * n_stage_ineq))],
        ]).build_csc();

        let q = Vector::zeros_generic(Dy::new(N * (ns + ni + n_stage_ineq)), U1);
        let mut l = Vector::zeros_generic(Dy::new(N * (ns + ni + 3 * n_stage_ineq)), U1);
        let mut u = Vector::zeros_generic(Dy::new(N * (ns + ni + 3 * n_stage_ineq)), U1);

        // Soft stage inequality default min and max constraints
        // Soft stage ineqs are disabled by default by forcing the slack variables to zero
        l.rows_mut(N * (ns + ni), N * n_stage_ineq * 2)
            .fill(NEG_INFINITY);
        u.rows_mut(N * (ns + ni), N * n_stage_ineq * 2)
            .fill(INFINITY);

        let settings = Settings::default()
            .verbose(log_enabled!(Debug))
            .polish(false)
            .eps_abs(1e-2);

        let problem = Problem::new(&P, q.as_slice(), &A, l.as_slice(), u.as_slice(), &settings);

        OsqpMpc {
            problem,
            N,
            n_stage_ineq,
            stage_ineq_lambda_max: 0.0,
            R: R.build_csc().to_dense(),
            q,
            Q_stage,
            R_stage_grad,
            A,
            l,
            u,
            A_blocks,
            B_blocks,
            stage_ineq_blocks,
            // Default input bounds
            u_min: Vector::from_element_generic(NI::name(), U1, NEG_INFINITY),
            u_max: Vector::from_element_generic(NI::name(), U1, INFINITY),
            u_delta_min: Vector::from_element_generic(NI::name(), U1, NEG_INFINITY),
            u_delta_max: Vector::from_element_generic(NI::name(), U1, INFINITY),
            x_mpc: Matrix::zeros_generic(NS::name(), Dy::new(N)),
            u_mpc: Matrix::zeros_generic(NI::name(), Dy::new(N)),
            u_prev: nalgebra::zero(),
        }
    }

    pub fn set_input_bounds(&mut self, u_min: Vector<NI>, u_max: Vector<NI>) {
        self.u_min = u_min;
        self.u_max = u_max;
    }

    pub fn set_input_delta_bounds(&mut self, u_delta_min: Vector<NI>, u_delta_max: Vector<NI>) {
        self.u_delta_min = u_delta_min;
        self.u_delta_max = u_delta_max;
    }

    pub fn set_model(
        &mut self,
        i: usize,
        A: &Matrix<NS, NS>,
        B: &Matrix<NS, NI>,
        x0: &Vector<NS>,
        u0: &Vector<NI>,
        x_target: &Vector<NS>,
        x_linear_penalty: &Vector<NS>,
    ) {
        assert!(i < self.N);

        if i > 0 {
            let A_block = &self.A_blocks[i - 1];
            self.A.set_block(A_block, A);
        }

        let B_block = &self.B_blocks[i];
        self.A.set_block(B_block, B);

        let ns = NS::dim();
        let ni = NI::dim();

        // Linear state penalty
        let q = &self.Q_stage * (x0 - x_target) + x_linear_penalty;
        self.q.rows_mut(i * ns, ns).copy_from(&q);

        // Calulcate lower and upper bounds
        let u_min = self.u_delta_min.zip_map(&(&self.u_min - u0), max);
        let u_max = self.u_delta_max.zip_map(&(&self.u_max - u0), min);

        let u_bounds_start = self.N * ns + i * ni;
        self.l.rows_mut(u_bounds_start, ni).copy_from(&u_min);
        self.u.rows_mut(u_bounds_start, ni).copy_from(&u_max);

        // Save x0 and u0
        self.x_mpc.column_mut(i).copy_from(x0);
        self.u_mpc.column_mut(i).copy_from(u0);
    }

    pub fn set_stage_inequality(
        &mut self,
        i: usize,
        j: usize,
        F: &Vector<NS>,
        min: float,
        max: float,
    ) {
        let N = self.N;
        let n_stage_ineq = self.n_stage_ineq;
        let ns = NS::dim();
        let ni = NI::dim();

        let block = &self.stage_ineq_blocks[n_stage_ineq * i + j];
        self.A.set_block(block, &F.transpose());

        self.l[N * (ni + ns) + n_stage_ineq * i + j] = min;
        self.u[N * (ni + ns + n_stage_ineq) + n_stage_ineq * i + j] = max;
    }

    pub fn solve(&mut self) -> Result<Solution<NS, NI>, ()> {
        let N = self.N;
        let ns = NS::dim();
        let ni = NI::dim();
        let n_stage_ineq = self.n_stage_ineq;

        // Set the u_0 values for the input constraints
        let guard = flame::start_guard("calculate input gradient penalty");
        {
            let u_0 = Matrix::<Dy, U1>::from_column_slice_generic(
                Dy::new(N * ni),
                U1,
                self.u_mpc.as_slice(),
            );
            self.R.mul_to(&u_0, &mut self.q.rows_mut(N * ns, N * ni));
            let mut top_q = self.q.fixed_rows_mut::<NI>(N * ns);
            top_q -= Matrix::from_diagonal(&self.R_stage_grad) * &self.u_prev;
        }
        guard.end();

        // Set upper and lower bounds for first input difference
        self.problem.update_lin_cost(self.q.as_slice());
        self.problem
            .update_bounds(self.l.as_slice(), self.u.as_slice());
        self.problem.update_A(&self.A);

        // Work around the lack of NLL in Rust
        // TODO: Remove this workaround once NLL is released (next three blocks)
        fn add_deltas<NS: DimName, NI: DimName>(
            x_mpc: &mut Matrix<NS, Dy>,
            u_mpc: &mut Matrix<NI, Dy>,
            x: &[float],
        ) where
            DefaultAllocator: Dims2<NS, NI>,
        {
            // Add the deltas to the solutions
            fn add_delta((x0, delta_x): (&mut float, &float)) {
                *x0 += *delta_x
            };
            let (ns, N) = x_mpc.shape();
            let ni = NI::dim();
            x_mpc.iter_mut().zip(&x[0..N * ns]).for_each(add_delta);
            u_mpc
                .iter_mut()
                .zip(&x[N * ns..N * (ni + ns)])
                .for_each(add_delta);
        };

        let primal_infeasible = match self.problem.solve() {
            Status::Solved(solution) | Status::SolvedInaccurate(solution) => {
                // Update soft constraint multipliers
                let lambda_max = solution
                    .y()
                    .iter()
                    .skip(N * (ns + ni))
                    .take(N * (2 * n_stage_ineq))
                    .fold(0.0, |acc, &v| max(acc, v.abs()));
                self.stage_ineq_lambda_max = max(lambda_max, self.stage_ineq_lambda_max);
                debug!(
                    "updating soft ineq penalty to: {}",
                    self.stage_ineq_lambda_max
                );

                // Update u_mpc and x_mpc
                add_deltas(&mut self.x_mpc, &mut self.u_mpc, solution.x());
                false
            }
            Status::MaxIterationsReached(_) => {
                println!("max iterations reached, retrying with soft constraints");
                true
            }
            Status::PrimalInfeasible(_) | Status::PrimalInfeasibleInaccurate(_) => {
                println!("primal problem infeasible!");
                true
            }
            Status::DualInfeasible(_) | Status::DualInfeasibleInaccurate(_) => {
                println!("dual problem infeasible!");
                return Err(());
            }
            _ => return Err(()),
        };

        if primal_infeasible {
            // Enable soft stage inequalities
            self.u
                .rows_mut(N * (ns + ni + n_stage_ineq * 2), N * n_stage_ineq)
                .fill(INFINITY);
            self.problem.update_upper_bound(self.u.as_slice());
            // Soft stage inequality variable linear penalty
            self.q
                .rows_mut(N * (ns + ni), N * n_stage_ineq)
                .fill(min(1e3, self.stage_ineq_lambda_max));
            self.problem.update_lin_cost(self.q.as_slice());

            // And disable them again
            self.u
                .rows_mut(N * (ns + ni + n_stage_ineq * 2), N * n_stage_ineq)
                .fill(0.0);
            // Soft stage inequality variable linear penalty
            self.q.rows_mut(N * (ns + ni), N * n_stage_ineq).fill(0.0);

            // Solve the soft constrained problem
            let solution = self.problem.solve().solution().ok_or(())?;

            // Update u_mpc and x_mpc
            add_deltas(&mut self.x_mpc, &mut self.u_mpc, solution.x());
        }

        // Validate input constraints
        for i in 0..N {
            let u = self.u_mpc.column(i);
            let a = u.iter()
                .zip(self.u_max.iter())
                .all(|(&val, &max)| val <= max + 0.1);
            let b = u.iter()
                .zip(self.u_min.iter())
                .all(|(&val, &min)| val >= min - 0.1);
            if !(a && b) {
                error!("{}", self.u_mpc);
                error!("u_min: {:?}", &self.l.as_slice()[N * ns..N * (ni + ns)]);
                error!("u_max: {:?}", &self.u.as_slice()[N * ns..N * (ni + ns)]);
                assert!(false, "u not within constraints");
            }
        }

        // Save input gradient
        self.u_prev.copy_from(&self.u_mpc.column(0));

        Ok(Solution {
            x: &self.x_mpc,
            u: &self.u_mpc,
            stage_ineq_violated: primal_infeasible,
        })
    }
}

pub struct Solution<'a, NS: DimName, NI: DimName> {
    pub x: &'a Matrix<NS, Dy>,
    pub u: &'a Matrix<NI, Dy>,
    pub stage_ineq_violated: bool,
}
