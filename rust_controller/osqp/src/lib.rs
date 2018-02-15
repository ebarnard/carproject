extern crate osqp as osqp_inner;
extern crate prelude;
extern crate sparse;

use flame;
use self::osqp_inner::Problem as InnerProblem;
pub use self::osqp_inner::*;

use prelude::*;

#[allow(non_snake_case)]
pub struct Problem {
    inner: InnerProblem,
}

#[allow(dead_code)]
impl Problem {
    #[allow(non_snake_case)]
    pub fn new(
        P: &sparse::CscMatrix,
        q: &[float],
        A: &sparse::CscMatrix,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Problem {
        let _guard = flame::start_guard("osqp setup");
        Problem {
            inner: InnerProblem::new(convert_sparse(P), q, convert_sparse(A), l, u, settings),
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

    pub fn update_upper_bound(&mut self, u: &[float]) {
        let _guard = flame::start_guard("osqp update_upper_bound");
        self.inner.update_upper_bound(u)
    }

    #[allow(non_snake_case)]
    pub fn update_P(&mut self, P: &sparse::CscMatrix) {
        let _guard = flame::start_guard("osqp update_P");
        self.inner.update_P(convert_sparse(P))
    }

    #[allow(non_snake_case)]
    pub fn update_A(&mut self, A: &sparse::CscMatrix) {
        let _guard = flame::start_guard("osqp update_A");
        self.inner.update_A(convert_sparse(A))
    }

    pub fn solve(&mut self) -> Status {
        let _guard = flame::start_guard("osqp solve");
        self.inner.solve()
    }
}

pub fn convert_sparse(this: &sparse::CscMatrix) -> CscMatrix {
    let (nrows, ncols) = this.shape();
    CscMatrix {
        nrows,
        ncols,
        indptr: (this.indptr()).into(),
        indices: (this.indices()).into(),
        data: (this.data()).into(),
    }
}
