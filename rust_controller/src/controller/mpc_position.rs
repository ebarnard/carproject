use flame;
use log::Level::Debug;
use nalgebra::{self, U2, VectorN};

use prelude::*;
use controller::{Controller, MpcBase};
use control_model::ControlModel;
use track::{CentrelineLookup, Track};

pub struct MpcPosition<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    base: MpcBase<M>,
    track: Track,
    lookup: CentrelineLookup,
}

impl<M: ControlModel> MpcPosition<M>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    pub fn new(model: &M, N: u32, track: &Track) -> MpcPosition<M> {
        let lookup = CentrelineLookup::from_track(track);

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
        let R = Matrix::from_diagonal(&R);

        let mut track_bounds_ineq_sparsity = VectorN::<bool, M::NS>::from_element(false);
        track_bounds_ineq_sparsity[0] = true;
        track_bounds_ineq_sparsity[1] = true;

        MpcPosition {
            base: MpcBase::new(model, N, Q, R, &[track_bounds_ineq_sparsity]),
            track: track.clone(),
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
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        p: &Vector<M::NP>,
    ) -> (&Matrix<M::NI, Dy>, &Matrix<M::NS, Dy>) {
        let v_target = dt * 2.0;
        let mut s_target = flame::span_of("centreline distance lookup", || {
            self.lookup.centreline_distance(x[0], x[1])
        });

        let lookup = &self.lookup;
        let track = &self.track;
        self.base.step(model, dt, x, p, |i, x_i, _u_i, mpc| {
            // Find track point
            s_target += v_target;
            let target = flame::span_of("centreline point lookup", || {
                track.nearest_centreline_point(s_target)
            });
            let theta = flame::span_of("theta calculation", || {
                float::atan2(target.dy_ds, target.dx_ds)
            });

            if log_enabled!(Debug) {
                debug!(
                    "target for ({}, {}, {}, {}) is: ({}, {}, {})",
                    x_i[0], x_i[1], x_i[2], x_i[3], target.x, target.y, theta
                );
                debug!(
                    "target distance {}",
                    float::hypot(x_i[0] - target.x, x_i[1] - target.y)
                );
            }

            let theta = phase_unwrap(x_i[2], theta);
            let mut x_target: Vector<M::NS> = nalgebra::zero();
            x_target[0] = target.x;
            x_target[1] = target.y;
            x_target[2] = theta;

            flame::span_of("track bounds ineq calculation", || {
                let s = lookup.centreline_distance(x_i[0], x_i[1]);
                let centreline_point = track.nearest_centreline_point(s);
                let a_i = centreline_point.a(x_i[0], x_i[1]);
                let J = centreline_point.jacobian(a_i);

                let mut delta_a_ineq: Vector<M::NS> = nalgebra::zero();
                delta_a_ineq
                    .fixed_rows_mut::<U2>(0)
                    .copy_from(&J.row(1).transpose());

                let a_max = centreline_point.track_width / 2.0;

                mpc.set_stage_inequality(i, 0, &delta_a_ineq, -a_max - a_i, a_max - a_i);
            });

            (x_target, nalgebra::zero())
        })
    }
}
