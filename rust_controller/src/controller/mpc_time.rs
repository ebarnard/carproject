use flame;
use nalgebra::{self, Dynamic, MatrixMN, U2, VectorN};

use prelude::*;
use controller::{Controller, OsqpMpc};
use control_model::{discretise, discretise_sparsity, ControlModel};
use track::{Centreline, CentrelineLookup, Track};

pub struct MpcTime<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    horizon: u32,
    u_mpc: Matrix<M::NI, Dynamic>,
    mpc: OsqpMpc<M::NS, M::NI>,
    centreline: Centreline,
    lookup: CentrelineLookup,
}

impl<M: ControlModel> MpcTime<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(model: &M, N: u32, track: &Track) -> MpcTime<M> {
        let centreline = Centreline::from_track(track);
        let lookup = CentrelineLookup::from_centreline(&centreline);

        // State penalties
        let Q: Matrix<M::NS, M::NS> = nalgebra::zero();

        // Input difference penalties
        // TODO: Support a configurable penalty multiplier
        let mut R: Vector<M::NI> = nalgebra::zero();
        R[0] = 15.0;
        R[1] = 15.0;

        // Some components of A and B will always be zero and can be excluded from the sparse
        // constraint matrix
        let (A_sparsity, B_sparsity) = model.linearise_sparsity();
        let (A_d_sparsity, B_d_sparsity) = discretise_sparsity(&A_sparsity, &B_sparsity);

        let mut track_bounds_ineq_sparsity = VectorN::<bool, M::NS>::from_element(false);
        track_bounds_ineq_sparsity[0] = true;
        track_bounds_ineq_sparsity[1] = true;

        let mpc = flame::span_of("osqp mpc create", || {
            let mut mpc = OsqpMpc::new(
                N as usize,
                Q,
                R,
                &A_d_sparsity,
                &B_d_sparsity,
                &[track_bounds_ineq_sparsity],
            );
            let (input_min, input_max) = model.input_bounds();
            mpc.set_input_bounds(input_min, input_max);
            mpc
        });

        MpcTime {
            horizon: N,
            u_mpc: MatrixMN::zeros_generic(<M::NI as DimName>::name(), Dynamic::new(N as usize)),
            mpc,
            centreline,
            lookup,
        }
    }
}

impl<M: ControlModel> Controller<M> for MpcTime<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NI>, Vector<M::NS>) {
        let _guard = flame::start_guard("controller step");
        let N = self.horizon as usize;

        let guard = flame::start_guard("mpc setup");

        let mut x_i = x.clone();

        for i in 0..N {
            let u_i = self.u_mpc.column(i).into_owned();

            // Linearise model around x_i-1 and u_i
            let (A_c, B_c) = flame::span_of("model linearise", || model.linearise(&x_i, &u_i, &p));
            let (A, B) = flame::span_of("model discretise", || discretise(dt, &A_c, &B_c));

            // Update state using nonlinear model
            x_i = flame::span_of("model integrate", || model.step(dt, &x_i, &u_i, &p));

            // Find centreline point
            let centreline = flame::span_of("centreline point lookup", || {
                let s = self.lookup.centreline_distance(x_i[0], x_i[1]);
                self.centreline.nearest_point(s)
            });

            let a_i = centreline.a(x_i[0], x_i[1]);
            let J = centreline.jacobian(a_i);

            // Minimise an approximate time penalty
            // Increasing speed is more important when minimising time if the car is travelling
            // slowly than if it is travelling fast.
            // TODO: Support a configurable penalty multiplier
            // TODO: Use a proper linearisation with a speed dependence
            let mut time_penalty: Vector<M::NS> = nalgebra::zero();
            time_penalty
                .fixed_rows_mut::<U2>(0)
                .copy_from(&-J.row(0).transpose());
            // Ensure we don't divide by zero
            time_penalty /= max(x_i[3], 0.01);

            let mut delta_a_ineq: Vector<M::NS> = nalgebra::zero();
            delta_a_ineq
                .fixed_rows_mut::<U2>(0)
                .copy_from(&J.row(1).transpose());

            let a_max = centreline.track_width / 2.0;

            // Give the values to the builder
            flame::span_of("update mpc matrices", || {
                self.mpc
                    .set_model(i, &A, &B, &x_i, &u_i, &nalgebra::zero(), &time_penalty);
                self.mpc
                    .set_stage_inequality(i, 0, &delta_a_ineq, -a_max - a_i, a_max - a_i);
            });
        }
        guard.end();

        let mpc = &mut self.mpc;
        let solution = flame::span_of("mpc solve", || mpc.solve());

        self.u_mpc
            .columns_mut(0, N - 1)
            .copy_from(&solution.u.columns(1, N - 1));
        self.u_mpc
            .column_mut(N - 1)
            .copy_from(&solution.u.column(N - 1));

        (
            solution.u.column(0).into_owned(),
            solution.x.column(0).into_owned(),
        )
    }
}
