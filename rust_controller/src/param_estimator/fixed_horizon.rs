use flame;
use log::LogLevel::Debug;
use nalgebra::{self, MatrixMN};
use osqp::{Settings, Workspace};
use sparse;

use prelude::*;
use control_model::{discretise, ControlModel};
use param_estimator::ParamEstimator;

pub struct FixedHorizon<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    workspace: Workspace,
    params: Vector<M::NP>,
    // Cost matrix for the state prediction errors
    Q: Matrix<M::NS, M::NS>,
    // QP hessian
    P: Matrix<M::NP, M::NP>,
    P_sparse: sparse::CSCMatrix,
    P_block: sparse::TrackedBlock<M::NP, M::NP>,
    // QP linear
    f: Vector<M::NP>,
    N: u32,
}

impl<M: ControlModel> FixedHorizon<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(delta_p_max: Vector<M::NP>, params: Vector<M::NP>) -> FixedHorizon<M> {
        // TODO: Choose this more carefully based on parameter scales.
        let Q = Matrix::<M::NS, M::NS>::identity();

        let P_sparsity = MatrixMN::<bool, M::NP, M::NP>::from_element(true);
        let (mut P_sparse, P_block) = sparse::Builder::tracked_sparse_block(&P_sparsity);
        let P_sparse = P_sparse.build_csc();
        let f: Vector<M::NP> = nalgebra::zero();

        // Maximum allowed delta_p deviation
        let A = sparse::Builder::eye(<M::NP as DimName>::dim()).build_csc();
        let l = -&delta_p_max;
        let l = l.as_slice();
        let u = delta_p_max.as_slice();

        let settings = Settings::default().verbose(log_enabled!(Debug));

        let workspace = Workspace::new(&P_sparse, f.as_slice(), &A, l, u, &settings);

        FixedHorizon {
            workspace,
            params,
            Q,
            P: nalgebra::zero(),
            P_sparse,
            P_block,
            f,
            N: 0,
        }
    }

    pub fn record_observation(
        &mut self,
        dt: float,
        x0: &Vector<M::NS>,
        u0: &Vector<M::NI>,
        x: &Vector<M::NS>,
    ) {
        // Linearise and discretise the vehicle model around its parameters
        let (A_c, _) = M::linearise(x0, u0, &self.params);
        let P_c = M::linearise_parameters(x0, u0, &self.params);
        let (_, P) = discretise(dt, &A_c, &P_c);

        // Predict the current state using x0 and the current model parameters
        let x_hat = M::step(dt, x0, u0, &self.params);

        // State prediction error
        let e = x - x_hat;

        // Add this set of ovservations to P and f
        let PtQ = P.transpose() * &self.Q;
        self.P += &PtQ * P;
        self.f -= PtQ * e;
        self.N += 1;
    }

    pub fn optimise(&mut self) -> &Vector<M::NP> {
        if self.N < 3 {
            return &self.params;
        }

        // Normalise scaling of P and q
        self.P *= 1.0 / self.N as float;
        self.f *= 1.0 / self.N as float;

        // TODO: Do we want a guassian prior on p_k-1?
        //self.P += 0.1 * ::nalgebra::MatrixMN::<float, M::NP, M::NP>::identity();

        // Update sparse representation of P
        self.P_sparse.set_block(&self.P_block, &self.P);

        self.workspace.update_lin_cost(self.f.as_slice());
        self.workspace.update_P(&self.P_sparse);

        let solution = self.workspace.solve();
        assert!(solution.status() == ::osqp::Status::Solved);

        self.params.iter_mut().zip(solution.x()).for_each(|(p0, delta_p)| *p0 += delta_p);

        // Reset problem
        self.N = 0;
        self.P.fill(0.0);
        self.f.fill(0.0);

        debug!("new params: {}", self.params);

        &self.params
    }
}

impl<M: ControlModel> ParamEstimator<M> for FixedHorizon<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn update(
        &mut self,
        dt: float,
        x0: &Vector<M::NS>,
        u: &Vector<M::NI>,
        x: &Vector<M::NS>,
    ) -> Vector<M::NP> {
        let _guard = flame::start_guard("parameter estimation");

        self.record_observation(dt, x0, u, x);
        self.optimise().clone()
    }
}
