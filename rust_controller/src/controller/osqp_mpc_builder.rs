use flame;
use itertools::{repeat_n, Itertools};
use log::LogLevel::Debug;
use nalgebra::{self, Dynamic as Dy, MatrixMN, U1};
use osqp::{Settings, Status, Workspace};

use prelude::*;
use sparse;

pub struct OsqpMpc<NS: DimName, NI: DimName>
where
    DefaultAllocator: Dims2<NS, NI>,
{
    workspace: Workspace,
    N: usize,
    // Objective:
    R: Matrix<Dy, Dy>,
    P: sparse::CSCMatrix,
    q: Vector<Dy>,
    Q_stage: Matrix<NS, NS>,
    R_stage_grad: Vector<NI>,
    // Inequalities:
    // N * ns state transition rows
    // N * ni input constraints
    A: sparse::CSCMatrix,
    l: Vector<Dy>,
    u: Vector<Dy>,
    A_blocks: Vec<sparse::TrackedBlock<NS, NS>>,
    B_blocks: Vec<sparse::TrackedBlock<NS, NI>>,
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
    ) -> OsqpMpc<NS, NI> {
        let N = N as usize;
        let ns = NS::dim();
        let ni = NI::dim();

        // Build P matrix
        let Q = sparse::Builder::block(&Q_stage);
        let Q = sparse::block_diag(&repeat_n(&Q, N).collect_vec());

        let R_grad = Matrix::from_diagonal(&R_stage_grad);
        let R_grad = sparse::Builder::block(&R_grad);
        let R_diag = sparse::block_diag(&repeat_n(&R_grad, N - 1).collect_vec());
        let R_main_diag = sparse::block_diag(&[&(&R_diag + &R_diag), &R_grad]);
        let R_min_diag = -R_diag;
        let R_sub_diag = sparse::bmat(&[
            &[None, Some(&sparse::Builder::zeros(ni, ni))],
            &[Some(&R_min_diag), None],
        ]);
        let R_super_diag = sparse::bmat(&[
            &[None, Some(&R_min_diag)],
            &[Some(&sparse::Builder::zeros(ni, ni)), None],
        ]);

        let mut R = R_main_diag + R_sub_diag + R_super_diag;
        let P = sparse::block_diag(&[&Q, &R]).build_csc();
        let R = R.build_csc().to_dense();

        // Build A matrix
        let (Ax, A_blocks): (Vec<_>, Vec<_>) = (1..N)
            .map(|_| sparse::Builder::tracked_sparse_block(A_sparsity))
            .unzip();
        let (Au, B_blocks): (Vec<_>, Vec<_>) = (0..N)
            .map(|_| sparse::Builder::tracked_sparse_block(B_sparsity))
            .unzip();

        let Ax = -sparse::Builder::eye(N * ns)
            + sparse::bmat(&[
                &[None, Some(sparse::Builder::zeros(ns, ns))],
                &[Some(sparse::block_diag(&Ax)), None],
            ]);
        let Au = sparse::block_diag(&Au);

        let A = sparse::bmat(&[
            // State evolution
            &[Some(Ax), Some(Au)],
            // Input absolute and delta constraints
            &[None, Some(sparse::Builder::eye(N * ni))],
        ]).build_csc();

        let q = Vector::zeros_generic(Dy::new(N * (ns + ni)), U1);
        let l = Vector::zeros_generic(Dy::new(N * (ns + ni)), U1);
        let u = Vector::zeros_generic(Dy::new(N * (ns + ni)), U1);

        let settings = Settings::default()
            .verbose(log_enabled!(Debug))
            .polish(false)
            .eps_abs(1e-2);

        let workspace = Workspace::new(&P, q.as_slice(), &A, l.as_slice(), u.as_slice(), &settings);

        OsqpMpc {
            workspace,
            N,
            R,
            P,
            q,
            Q_stage,
            R_stage_grad,
            A,
            l,
            u,
            A_blocks,
            B_blocks,
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
    ) {
        assert!(i < self.N);
        let i = i as usize;

        if i > 0 {
            let A_block = &self.A_blocks[i - 1];
            self.A.set_block(A_block, A);
        }

        let B_block = &self.B_blocks[i];
        self.A.set_block(B_block, B);

        let ns = NS::dim();
        let ni = NI::dim();
        let N = self.N as usize;

        // Linear state penalty
        self.q
            .rows_mut(i * ns, ns)
            .copy_from(&(&self.Q_stage * (x0 - x_target)));

        // Calulcate lower and upper bounds
        let u_min = self.u_delta_min
            .zip_map(&(&self.u_min - u0), |a, b| max(a, b));
        let u_max = self.u_delta_max
            .zip_map(&(&self.u_max - u0), |a, b| min(a, b));

        let intput_bounds_start = N * ns + i * ni;
        self.l.rows_mut(intput_bounds_start, ni).copy_from(&u_min);
        self.u.rows_mut(intput_bounds_start, ni).copy_from(&u_max);

        // Save x0 and u0
        self.x_mpc.column_mut(i).copy_from(&x0);
        self.u_mpc.column_mut(i).copy_from(&u0);
    }

    pub fn solve(&mut self) -> Solution<NS, NI> {
        let N = self.N;
        let ns = NS::dim();
        let ni = NI::dim();

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
        self.workspace.update_lin_cost(self.q.as_slice());
        self.workspace
            .update_bounds(self.l.as_slice(), self.u.as_slice());
        self.workspace.update_A(&self.A);

        let solution = self.workspace.solve();

        match solution.status() {
            Status::Solved => (),
            _ => panic!("solver failed"),
        }

        // Add the deltas to the solutions
        fn add_delta((x0, delta_x): (&mut float, &float)) {
            *x0 += *delta_x
        };
        self.x_mpc
            .iter_mut()
            .zip(&solution.x()[0..N * ns])
            .for_each(add_delta);
        self.u_mpc
            .iter_mut()
            .zip(&solution.x()[N * ns..N * (ni + ns)])
            .for_each(add_delta);

        // Validate constraints
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
                error!("delta_u: {:?}", &solution.x()[N * ns..N * (ni + ns)]);
                error!("u_min: {:?}", &self.l.as_slice()[N * ns..N * (ni + ns)]);
                error!("u_max: {:?}", &self.u.as_slice()[N * ns..N * (ni + ns)]);
                assert!(false, "u not within constraints");
            }
        }

        // Save input gradient
        self.u_prev.copy_from(&self.u_mpc.column(0));

        Solution {
            x: &self.x_mpc,
            u: &self.u_mpc,
        }
    }
}

pub struct Solution<'a, NS: DimName, NI: DimName> {
    pub x: &'a Matrix<NS, Dy>,
    pub u: &'a Matrix<NI, Dy>,
}
