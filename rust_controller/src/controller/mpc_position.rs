use flame;
use log::LogLevel::Debug;
use nalgebra::{self, DefaultAllocator, DimName, Dynamic, MatrixMN};

use prelude::*;
use controller::{Controller, OsqpMpc};
use control_model::{discretise, discretise_nonzero_mask, ControlModel};
use track::{Centreline, CentrelineLookup, Track};

pub struct MpcPosition<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    horizon: u32,
    u_mpc: Matrix<M::NI, Dynamic>,
    mpc: OsqpMpc<M::NS, M::NI>,
    centreline: Centreline,
    lookup: CentrelineLookup,
}

impl<M: ControlModel> MpcPosition<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(N: u32, track: &Track) -> MpcPosition<M> {
        let centreline = Centreline::from_track(track);
        let lookup = CentrelineLookup::from_centreline(&centreline);

        // State penalties
        let mut Q: Vector<M::NS> = nalgebra::zero();
        Q[0] = 20.0;
        Q[1] = 20.0;
        Q[2] = 3.0;
        let Q = Matrix::from_diagonal(&Q);

        // Input difference penalties
        let mut R: Vector<M::NI> = nalgebra::zero();
        R[0] = 0.1;
        R[1] = 1.0;

        // Some components of A and B will always be zero and can be excluded from the sparse
        // constraint matrix
        let (A_sparsity, B_sparsity) = M::linearise_nonzero_mask();
        let (A_d_sparsity, B_d_sparsity) = discretise_nonzero_mask(&A_sparsity, &B_sparsity);

        let mpc = flame::span_of("osqp mpc create", || {
            let mut mpc = OsqpMpc::new(N as usize, Q, R, &A_d_sparsity, &B_d_sparsity);
            let (input_min, input_max) = M::input_bounds();
            mpc.set_input_bounds(input_min, input_max);
            mpc
        });

        MpcPosition {
            horizon: N,
            u_mpc: MatrixMN::zeros_generic(<M::NI as DimName>::name(), Dynamic::new(N as usize)),
            mpc,
            centreline,
            lookup,
        }
    }
}

impl<M: ControlModel> Controller<M> for MpcPosition<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        dt: float,
        x: &Vector<M::NS>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NI>, Vector<M::NS>) {
        let _guard = flame::start_guard("controller step");
        let N = self.horizon as usize;

        let guard = flame::start_guard("mpc setup");

        let mut centreline_distance = flame::span_of(
            "centreline distance lookup",
            || self.lookup.centreline_distance(x[0], x[1]),
        );

        let s_target = dt * 2.0;
        centreline_distance += 2.0 * s_target;

        let mut x_i = x.clone();

        for i in 0..N {
            let u_i = self.u_mpc.column(i).into_owned();

            // Linearise model around x_i-1 and u_i
            let (A_i_c, B_i_c) = flame::span_of("model linearise", || M::linearise(&x_i, &u_i, &p));
            let (A_i, B_i) = flame::span_of("model discretise", || discretise(dt, &A_i_c, &B_i_c));

            // Update state using nonlinear model
            x_i = flame::span_of("model integrate", || M::step(dt, &x_i, &u_i, &p));

            // Find centreline point
            centreline_distance += s_target;
            let target = flame::span_of(
                "centreline point lookup",
                || self.centreline.nearest_point(centreline_distance),
            );
            let theta =
                flame::span_of("theta calculation", || float::atan2(target.dy_ds, target.dx_ds));

            if log_enabled!(Debug) {
                debug!(
                    "target for ({}, {}, {}, {}) is: ({}, {}, {})",
                    x_i[0],
                    x_i[1],
                    x_i[2],
                    x_i[3],
                    target.x,
                    target.y,
                    theta
                );
                debug!("target distance {}", float::hypot(x_i[0] - target.x, x_i[1] - target.y));
            }

            let theta = phase_unwrap(x_i[2], theta);
            let mut x_target: Vector<M::NS> = nalgebra::zero();
            x_target[0] = target.x;
            x_target[1] = target.y;
            x_target[2] = theta;

            // Give the values to the builder
            flame::span_of(
                "update mpc matrices",
                || self.mpc.set_model(i, &A_i, &B_i, &x_i, &u_i, &x_target),
            );
        }
        guard.end();

        let mpc = &mut self.mpc;
        let solution = flame::span_of("mpc solve", || mpc.solve());

        self.u_mpc.columns_mut(0, N - 1).copy_from(&solution.u.columns(1, N - 1));
        self.u_mpc.column_mut(N - 1).copy_from(&solution.u.column(N - 1));

        (solution.u.column(0).into_owned(), solution.x.column(0).into_owned())
    }
}
