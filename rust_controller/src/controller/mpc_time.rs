use flame;
use nalgebra::{self, U2, VectorN};

use prelude::*;
use controller::{Controller, MpcBase};
use control_model::ControlModel;
use track::{Centreline, CentrelineLookup, Track};

pub struct MpcTime<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    base: MpcBase<M>,
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

        let mut track_bounds_ineq_sparsity = VectorN::<bool, M::NS>::from_element(false);
        track_bounds_ineq_sparsity[0] = true;
        track_bounds_ineq_sparsity[1] = true;

        MpcTime {
            base: MpcBase::new(model, N, Q, R, &[track_bounds_ineq_sparsity]),
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
        let lookup = &self.lookup;
        let centreline = &self.centreline;
        self.base.step(model, dt, x, p, |i, x_i, _u_i, mpc| {
            // Find centreline point
            let centreline = flame::span_of("centreline point lookup", || {
                let s = lookup.centreline_distance(x_i[0], x_i[1]);
                centreline.nearest_point(s)
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
            flame::span_of("update mpc inequalities", || {
                mpc.set_stage_inequality(i, 0, &delta_a_ineq, -a_max - a_i, a_max - a_i);
            });

            (nalgebra::zero(), time_penalty)
        })
    }
}
