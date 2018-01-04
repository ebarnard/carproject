extern crate osqp as osqp_inner;

use flame;
use self::osqp_inner::Workspace as InnerWorkspace;
pub use self::osqp_inner::*;

use prelude::*;

#[allow(non_snake_case)]
pub struct Workspace {
    inner: InnerWorkspace,
}

#[allow(dead_code)]
impl Workspace {
    #[allow(non_snake_case)]
    pub fn new(
        P: &sparse::CSCMatrix,
        q: &[float],
        A: &sparse::CSCMatrix,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Workspace {
        let _guard = flame::start_guard("osqp setup");
        Workspace {
            inner: InnerWorkspace::new(convert_sparse(P), q, convert_sparse(A), l, u, settings),
        }
    }

    pub fn update_lin_cost(&mut self, q: &[float]) {
        let _guard = flame::start_guard("osqp update_lin_cost");
        self.inner.update_lin_cost(q)
    }

    pub fn update_bounds(&mut self, l: &[float], u: &[float]) {
        let _guard = flame::start_guard("osqp update_bounds");
        self.inner.update_bounds(l, u)
    }

    #[allow(non_snake_case)]
    pub fn update_P(&mut self, P: &sparse::CSCMatrix) {
        let _guard = flame::start_guard("osqp update_P");
        self.inner.update_P(convert_sparse(P))
    }

    #[allow(non_snake_case)]
    pub fn update_A(&mut self, A: &sparse::CSCMatrix) {
        let _guard = flame::start_guard("osqp update_A");
        self.inner.update_A(convert_sparse(A))
    }

    pub fn solve(&mut self) -> Solution {
        let _guard = flame::start_guard("osqp solve");
        self.inner.solve()
    }
}

use sparse;

pub fn convert_sparse(this: &sparse::CSCMatrix) -> CscMatrix {
    let (nrows, ncols) = this.shape();
    CscMatrix {
        nrows,
        ncols,
        indptr: (this.indptr()).into(),
        indices: (this.indices()).into(),
        data: (this.data()).into(),
    }
}
