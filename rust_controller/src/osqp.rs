extern crate osqp as osqp_inner;

use flame;
use self::osqp_inner::Workspace as InnerWorkspace;
pub use self::osqp_inner::*;

use prelude::*;

#[allow(non_snake_case)]
pub struct Workspace {
    inner: InnerWorkspace,
}

impl Workspace {
    #[allow(non_snake_case)]
    pub fn new(
        P: CscMatrix,
        q: &[float],
        A: CscMatrix,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Workspace {
        let _guard = flame::start_guard("osqp setup");
        Workspace {
            inner: InnerWorkspace::new(P, q, A, l, u, settings),
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

    pub fn update_lower_bound(&mut self, l: &[float]) {
        let _guard = flame::start_guard("osqp update_lower_bound");
        self.inner.update_lower_bound(l)
    }

    pub fn update_upper_bound(&mut self, u: &[float]) {
        let _guard = flame::start_guard("osqp update_upper_bound");
        self.inner.update_upper_bound(u)
    }

    pub fn warm_start(&mut self, x: &[float], y: &[float]) {
        let _guard = flame::start_guard("osqp warm_start");
        self.inner.warm_start(x, y)
    }

    pub fn warm_start_x(&mut self, x: &[float]) {
        let _guard = flame::start_guard("osqp warm_start_x");
        self.inner.warm_start_x(x)
    }

    pub fn warm_start_y(&mut self, y: &[float]) {
        let _guard = flame::start_guard("osqp warm_start_y");
        self.inner.warm_start_y(y)
    }

    #[allow(non_snake_case)]
    pub fn update_P(&mut self, P_data: &[float]) {
        let _guard = flame::start_guard("osqp update_P");
        self.inner.update_P(P_data)
    }

    #[allow(non_snake_case)]
    pub fn update_A(&mut self, A_data: &[float]) {
        let _guard = flame::start_guard("osqp update_A");
        self.inner.update_A(A_data)
    }

    #[allow(non_snake_case)]
    pub fn update_P_A(&mut self, P_data: &[float], A_data: &[float]) {
        let _guard = flame::start_guard("osqp update_P_A");
        self.inner.update_P_A(P_data, A_data)
    }

    pub fn solve<'a>(&'a mut self) -> Solution<'a> {
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
