use flame;
use log::LogLevel::Debug;
use nalgebra::{self, DefaultAllocator, DimName, Dynamic, MatrixMN};

use prelude::*;
use controller::{Control, Controller, OsqpMpc, State};
use control_model::{discretise, discretise_nonzero_mask, ControlModel};
use track::{Centreline, CentrelineLookup, Track};

pub struct MpcPosition<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    horizon: u32,
    model: M,
    u_mpc: Matrix<M::NI, Dynamic>,
    mpc: OsqpMpc<M::NS, M::NI>,
    centreline: Centreline,
    lookup: CentrelineLookup,
}

impl<M: ControlModel> MpcPosition<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(N: u32, model: M, track: &Track) -> MpcPosition<M> {
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
        let (A_sparsity, B_sparsity) = model.linearise_nonzero_mask();
        let (A_d_sparsity, B_d_sparsity) = discretise_nonzero_mask(&A_sparsity, &B_sparsity);

        let mpc = flame::span_of("osqp mpc create", || {
            let mut mpc = OsqpMpc::new(N as usize, Q, R, &A_d_sparsity, &B_d_sparsity);
            let (input_min, input_max) = model.input_bounds();
            mpc.set_input_bounds(input_min, input_max);
            mpc
        });

        MpcPosition {
            horizon: N,
            model,
            u_mpc: MatrixMN::zeros_generic(<M::NI as DimName>::name(), Dynamic::new(N as usize)),
            mpc,
            centreline,
            lookup,
        }
    }
}

impl<M: ControlModel> Controller for MpcPosition<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(&mut self, dt: float, state: &State, params: &[float]) -> (Control, State) {
        let _guard = flame::start_guard("controller step");
        let N = self.horizon as usize;

        let guard = flame::start_guard("mpc setup");

        let mut p: Vector<M::NP> = ::nalgebra::zero();
        p.as_mut_slice().copy_from_slice(params);

        let x_0 = self.model.x_from_state(state);
        let mut centreline_distance = flame::span_of(
            "centreline distance lookup",
            || self.lookup.centreline_distance(x_0[0], x_0[1]),
        );

        let s_target = dt * 2.0;
        centreline_distance += 2.0 * s_target;

        let mut x_i = x_0;

        for i in 0..N {
            let u_i = self.u_mpc.column(i).into_owned();

            // Linearise model around x_i-1 and u_i
            let (A_i_c, B_i_c) =
                flame::span_of("model linearise", || self.model.linearise(&x_i, &u_i, &p));
            let (A_i, B_i) = flame::span_of("model discretise", || discretise(dt, &A_i_c, &B_i_c));

            // Update state using nonlinear model
            x_i = flame::span_of("model integrate", || self.model.step(dt, &x_i, &u_i, &p));

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

        (
            self.model.u_to_control(&solution.u.column(0).into_owned()),
            self.model.x_to_state(&solution.x.column(0).into_owned()),
        )
    }
}

fn phase_unwrap(a: float, mut b: float) -> float {
    if a.is_infinite() || b.is_infinite() {
        return b;
    }
    while b > a + PI {
        b -= 2.0 * PI;
    }
    while b < a - PI {
        b += 2.0 * PI;
    }
    b
}
