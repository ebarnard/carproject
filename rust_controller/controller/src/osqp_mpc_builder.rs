use log::Level::Debug;
use nalgebra::{Dynamic as Dy, MatrixMN, U1};
use osqp::{Problem, Settings, Status};
use std::iter::{once, repeat};
use std::time::{Duration, Instant};

use prelude::*;
use sparse;

pub struct OsqpMpc<NS: DimName, NI: DimName>
where
    DefaultAllocator: Dims2<NS, NI>,
{
    problem: Problem,
    N: usize,
    n_stage_ineq: usize,
    n_virtual_states: usize,
    stage_ineq_lambda_max: float,
    // Objective:
    q: Vector<Dy>,
    Q_stage: Matrix<Dy, Dy>,
    Q_terminal: Matrix<Dy, Dy>,
    R_stage_grad: Matrix<NI, NI>,
    // Inequalities:
    // N * ns state transition rows
    // N * ni input constraints
    A: sparse::CscMatrix,
    l: Vector<Dy>,
    u: Vector<Dy>,
    A_blocks: Vec<sparse::BlockRef<NS, NS>>,
    B_blocks: Vec<sparse::BlockRef<NS, NI>>,
    stage_ineq_blocks: Vec<sparse::BlockRef<Dy, Dy>>,
    u_min: Vector<NI>,
    u_max: Vector<NI>,
    u_delta_min: Vector<NI>,
    u_delta_max: Vector<NI>,
    // Optimisation results
    x_mpc: Matrix<NS, Dy>,
    u_mpc: Matrix<NI, Dy>,
}

// x_i = A_i * delta_x_i-1 + B_i * delta_u_i-1 + x0_i
pub struct MpcStage<'a, NS: 'a + DimName, NI: 'a + DimName>
where
    DefaultAllocator: Dims2<NS, NI>,
{
    pub A: &'a Matrix<NS, NS>,
    pub B: &'a Matrix<NS, NI>,
    pub x0: &'a Vector<NS>,
    pub u0: &'a Vector<NI>,
    pub x_target: &'a Vector<Dy>,
    pub x_linear_penalty: &'a Vector<Dy>,
    pub stage_ineq: &'a Matrix<Dy, Dy>,
    pub stage_ineq_min: &'a Vector<Dy>,
    pub stage_ineq_max: &'a Vector<Dy>,
}

// TODO: Use const integer types instead of Dy once available.
impl<NS: DimName, NI: DimName> OsqpMpc<NS, NI>
where
    DefaultAllocator: Dims2<NS, NI>,
{
    pub fn new(
        N: usize,
        Q_stage: Matrix<Dy, Dy>,
        Q_terminal: Matrix<Dy, Dy>,
        R_stage_grad: Matrix<NI, NI>,
        A_sparsity: &MatrixMN<bool, NS, NS>,
        B_sparsity: &MatrixMN<bool, NS, NI>,
        stage_ineq_sparsity: &MatrixMN<bool, Dy, Dy>,
    ) -> OsqpMpc<NS, NI> {
        let N = N as usize;
        let ns = NS::dim();
        let ni = NI::dim();
        let (n_stage_ineq, n_total_states) = stage_ineq_sparsity.shape();
        assert!(n_total_states >= ns);
        let n_virtual_states = n_total_states - ns;

        assert_eq!(Q_stage.shape(), (n_total_states, n_total_states));
        assert_eq!(Q_stage.shape(), Q_terminal.shape());
        assert_eq!(stage_ineq_sparsity.shape().1, n_total_states);

        // Build state quadratic penalty
        let Q = sparse::block_diag(&repeat(&sparse::block(&Q_stage))
            .take(N - 1)
            .chain(once(&sparse::block(&Q_terminal)))
            .collect::<Vec<_>>());

        // Build input delta quadratic penalty
        let R_grad = sparse::block(&R_stage_grad);
        let R_grad = sparse::block_diag(&repeat(&R_grad).take(N).collect::<Vec<_>>());

        // Build penalty matrix P
        let P = sparse::block_diag(&[
            &Q,
            // Absolute inputs have no quadratic cost
            &sparse::zeros(N * ni, N * ni),
            &R_grad,
            // Soft stage inequality penalty variable has no quadratic cost
            &sparse::zeros(N * n_stage_ineq, N * n_stage_ineq),
        ]).build_csc();

        // Build state evolution matrices
        let (Ax, A_blocks): (Vec<_>, Vec<_>) = (1..N)
            .map(|_| {
                let (A, A_block) = sparse::block_mut(A_sparsity);
                (
                    sparse::hstack(&[A, sparse::zeros(ns, n_virtual_states)]),
                    A_block,
                )
            })
            .unzip();
        let (Au, B_blocks): (Vec<_>, Vec<_>) =
            (0..N).map(|_| sparse::block_mut(B_sparsity)).unzip();
        let A_evolution_diag: Vec<_> = (0..N)
            .map(|_| sparse::hstack(&[sparse::eye(ns), sparse::zeros(ns, n_virtual_states)]))
            .collect();

        let Ax = -sparse::block_diag(&A_evolution_diag)
            + sparse::bmat(&[
                &[None, Some(sparse::zeros(ns, ns + n_virtual_states))],
                &[Some(sparse::block_diag(&Ax)), None],
            ]);
        let Au = sparse::block_diag(&Au);

        // Build input to input difference equality matrix
        let A_R_grad = sparse::diags(N * ni, &[1.0, -1.0], &[0, -(ni as isize)]);

        // Build stage soft inequality matrix
        // Soft inequalities are formulated as:
        // 0         < b_i           < INFINITY
        // min       < F_i * x + b_i < INFINITY
        // -INFINITY < F_i * x - b_i < max
        let (stage_ineqs, stage_ineq_blocks): (Vec<_>, Vec<_>) = (0..N)
            .map(|_| sparse::block_mut(stage_ineq_sparsity))
            .unzip();
        let stage_ineqs = sparse::block_diag(&stage_ineqs);

        // Build constraint matrix A
        let A = sparse::bmat(&[
            // State evolution
            &[Some(&Ax), Some(&Au), None, None],
            // Input gradient equalities
            &[None, Some(&A_R_grad), Some(&-sparse::eye(N * ni)), None],
            // Input absolute and delta constraints
            &[None, Some(&sparse::eye(N * ni)), None, None],
            // Soft stage inequality min constraints
            &[
                Some(&stage_ineqs),
                None,
                None,
                Some(&sparse::eye(N * n_stage_ineq)),
            ],
            // Soft stage inequality max constraints
            &[
                Some(&stage_ineqs),
                None,
                None,
                Some(&-sparse::eye(N * n_stage_ineq)),
            ],
            // Soft stage inequality penalty variable is positive constraint
            &[None, None, None, Some(&sparse::eye(N * n_stage_ineq))],
        ]).build_csc();

        let q = Vector::zeros_generic(
            Dy::new(N * (ns + n_virtual_states + 2 * ni + n_stage_ineq)),
            U1,
        );
        let mut l = Vector::zeros_generic(Dy::new(N * (ns + 2 * ni + 3 * n_stage_ineq)), U1);
        let mut u = Vector::zeros_generic(Dy::new(N * (ns + 2 * ni + 3 * n_stage_ineq)), U1);

        // Soft stage inequality default min and max constraints
        l.rows_mut(N * (ns + 2 * ni), N * n_stage_ineq * 2)
            .fill(NEG_INFINITY);
        u.rows_mut(N * (ns + 2 * ni), N * n_stage_ineq * 2)
            .fill(INFINITY);

        let settings = Settings::default()
            .verbose(log_enabled!(Debug))
            .polish(false)
            .eps_abs(1e-2)
            .max_iter(250);

        let problem = Problem::new(&P, q.as_slice(), &A, l.as_slice(), u.as_slice(), &settings);

        OsqpMpc {
            problem,
            N,
            n_stage_ineq,
            n_virtual_states,
            stage_ineq_lambda_max: 0.0,
            q,
            Q_stage,
            Q_terminal,
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

    pub fn set_model(&mut self, i: usize, s: &MpcStage<NS, NI>) {
        let ns = NS::dim();
        let ni = NI::dim();
        let N = self.N;
        let n_stage_ineq = self.n_stage_ineq;
        let n_virtual_states = self.n_virtual_states;

        assert!(i < N);
        assert_eq!(s.x_target.shape().0, ns + n_virtual_states);
        assert_eq!(s.x_linear_penalty.shape().0, ns + n_virtual_states);

        if i > 0 {
            let A_block = &self.A_blocks[i - 1];
            self.A.set_block(A_block, s.A);
        }

        let B_block = &self.B_blocks[i];
        self.A.set_block(B_block, s.B);

        // Linear state penalty
        let Q = if i == N - 1 {
            &self.Q_terminal
        } else {
            &self.Q_stage
        };

        let mut x0_virtual = Vector::zeros_generic(Dy::new(ns + n_virtual_states), U1);
        x0_virtual.fixed_rows_mut::<NS>(0).copy_from(s.x0);
        let q = Q * (x0_virtual - s.x_target) + s.x_linear_penalty;
        self.q
            .rows_mut(i * (ns + n_virtual_states), ns + n_virtual_states)
            .copy_from(&q);

        // Calulcate lower and upper bounds
        let u_min = self.u_delta_min.zip_map(&(&self.u_min - s.u0), max);
        let u_max = self.u_delta_max.zip_map(&(&self.u_max - s.u0), min);

        let u_bounds_start = self.N * (ns + ni) + i * ni;
        self.l
            .fixed_rows_mut::<NI>(u_bounds_start)
            .copy_from(&u_min);
        self.u
            .fixed_rows_mut::<NI>(u_bounds_start)
            .copy_from(&u_max);

        // Save x0 and u0
        self.x_mpc.column_mut(i).copy_from(s.x0);
        self.u_mpc.column_mut(i).copy_from(s.u0);

        // Set stage ineqs

        let block = &self.stage_ineq_blocks[i];
        self.A.set_block(block, &s.stage_ineq);

        self.l
            .rows_mut(N * (ns + 2 * ni) + n_stage_ineq * i, n_stage_ineq)
            .copy_from(s.stage_ineq_min);
        self.u
            .rows_mut(N * (ns + 2 * ni) + n_stage_ineq * i, n_stage_ineq)
            .copy_from(s.stage_ineq_max);
    }

    pub fn solve(
        &mut self,
        u_prev: Vector<NI>,
        time_limit: Duration,
    ) -> Result<Solution<NS, NI>, ()> {
        let solve_start = Instant::now();

        let N = self.N;
        let ns = NS::dim();
        let n_virtual_states = self.n_virtual_states;
        let ni = NI::dim();
        let n_stage_ineq = self.n_stage_ineq;

        // Set the u_0 values for the input constraints
        let mut u0_i_minus_1 = u_prev;
        for i in 0..N {
            let u0_i = self.u_mpc.column(i);
            let u0_i_grad = u0_i - u0_i_minus_1;
            u0_i_minus_1 = u0_i.into_owned();

            self.q
                .fixed_rows_mut::<NI>(N * (ns + n_virtual_states + ni) + i * ni)
                .copy_from(&(&self.R_stage_grad * u0_i_grad));
        }

        // Set upper and lower bounds for first input difference
        self.problem.update_lin_cost(self.q.as_slice());

        for (i, (&l, &u)) in self.l.as_slice().iter().zip(self.u.as_slice()).enumerate() {
            if l > u {
                println!("l > u at i={} l={} u={}", i, l, u);
            }
        }

        self.problem
            .update_bounds(self.l.as_slice(), self.u.as_slice());
        self.problem.update_A(&self.A);

        let solve_time_limit = time_limit
            .checked_sub(solve_start.elapsed())
            .unwrap_or_default();
        self.problem.update_time_limit(Some(solve_time_limit));

        // Work around the lack of NLL in Rust
        // TODO: Remove this workaround once NLL is released (next three blocks)
        fn add_deltas<NS: DimName, NI: DimName>(
            x_mpc: &mut Matrix<NS, Dy>,
            u_mpc: &mut Matrix<NI, Dy>,
            x: &[float],
            n_virtual_states: usize,
        ) where
            DefaultAllocator: Dims2<NS, NI>,
        {
            // Add the deltas to the solutions
            fn add_delta((x0, delta_x): (&mut float, &float)) {
                *x0 += *delta_x
            };
            let (ns, N) = x_mpc.shape();
            let ni = NI::dim();
            for (i, delta_x) in x.chunks(ns + n_virtual_states).take(N).enumerate() {
                x_mpc
                    .column_mut(i)
                    .iter_mut()
                    .zip(delta_x)
                    .for_each(add_delta);
            }
            for (i, delta_u) in (&x[N * (ns + n_virtual_states)..])
                .chunks(ni)
                .take(N)
                .enumerate()
            {
                u_mpc
                    .column_mut(i)
                    .iter_mut()
                    .zip(delta_u)
                    .for_each(add_delta);
            }
        };

        let primal_infeasible = match self.problem.solve() {
            Status::Solved(solution)
            | Status::SolvedInaccurate(solution)
            | Status::MaxIterationsReached(solution)
            | Status::TimeLimitReached(solution) => {
                // Update soft constraint multipliers
                let lambda_max = solution
                    .y()
                    .iter()
                    .skip(N * (ns + 2 * ni))
                    .take(N * (2 * n_stage_ineq))
                    .fold(0.0, |acc, &v| max(acc, v.abs()));
                self.stage_ineq_lambda_max = max(lambda_max, self.stage_ineq_lambda_max);
                debug!(
                    "updating soft ineq penalty to: {}",
                    self.stage_ineq_lambda_max
                );

                // Update u_mpc and x_mpc
                add_deltas(
                    &mut self.x_mpc,
                    &mut self.u_mpc,
                    solution.x(),
                    n_virtual_states,
                );
                false
            }
            Status::PrimalInfeasible(_) | Status::PrimalInfeasibleInaccurate(_) => {
                println!("hard constrained primal problem infeasible!");
                true
            }
            Status::DualInfeasible(_) | Status::DualInfeasibleInaccurate(_) => {
                println!("hard constrained dual problem infeasible!");
                return Err(());
            }
            _ => return Err(()),
        };

        if primal_infeasible {
            // Enable soft stage inequalities
            self.u
                .rows_mut(N * (ns + 2 * ni + 2 * n_stage_ineq), N * n_stage_ineq)
                .fill(INFINITY);
            self.problem.update_upper_bound(self.u.as_slice());
            // Soft stage inequality variable linear penalty
            self.q
                .rows_mut(N * (ns + n_virtual_states + 2 * ni), N * n_stage_ineq)
                .fill(min(1e3, self.stage_ineq_lambda_max));
            self.problem.update_lin_cost(self.q.as_slice());

            // And disable them again
            self.u
                .rows_mut(N * (ns + 2 * ni + 2 * n_stage_ineq), N * n_stage_ineq)
                .fill(0.0);
            // Soft stage inequality variable linear penalty
            self.q
                .rows_mut(N * (ns + n_virtual_states + 2 * ni), N * n_stage_ineq)
                .fill(0.0);

            // Solve the soft constrained problem
            let solution = match self.problem.solve() {
                Status::Solved(solution) | Status::SolvedInaccurate(solution) => solution,
                Status::MaxIterationsReached(solution) => {
                    println!("max iterations reached on soft constrained problem");
                    solution
                }
                Status::TimeLimitReached(solution) => {
                    println!("time limit reached on soft constrained problem");
                    solution
                }
                Status::PrimalInfeasible(_) | Status::PrimalInfeasibleInaccurate(_) => {
                    println!("soft constrained primal problem infeasible!");
                    return Err(());
                }
                Status::DualInfeasible(_) | Status::DualInfeasibleInaccurate(_) => {
                    println!("soft constrained dual problem infeasible!");
                    return Err(());
                }
                _ => return Err(()),
            };

            // Update u_mpc and x_mpc
            add_deltas(
                &mut self.x_mpc,
                &mut self.u_mpc,
                solution.x(),
                n_virtual_states,
            );
        }

        // Validate input constraints
        let mut u_not_within_constraints = false;
        for i in 0..N {
            let mut u = self.u_mpc.column_mut(i);
            for ((u, &u_min), &u_max) in u.iter_mut().zip(&self.u_min).zip(&self.u_max) {
                if *u < u_min - 1e-2 {
                    *u = u_min;
                    u_not_within_constraints = true;
                } else if *u > u_max + 1e-2 {
                    *u = u_max;
                    u_not_within_constraints = true;
                }
            }
        }
        if u_not_within_constraints {
            println!("u not within constraints");
        }

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
