use flame;
use nalgebra::{self, MatrixMN, U1, U2};
use std::sync::Arc;
use std::time::Duration;

use control_model::ControlModel;
use prelude::*;
use track::TrackAndLookup;
use {Controller, MpcBase};

pub struct MpcDistance<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    base: MpcBase<M>,
    track: Arc<TrackAndLookup>,
}

impl<M: ControlModel> Controller<M> for MpcDistance<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn new(model: &M, N: u32, track: &Arc<TrackAndLookup>) -> MpcDistance<M> {
        let ns = M::NS::dim();
        let ns_virtual_dim = Dy::new(M::NS::dim() + 1);

        // State penalties
        let mut Q = Matrix::zeros_generic(ns_virtual_dim, ns_virtual_dim);
        Q[(ns, ns)] = 20.0;

        let mut Q_terminal = Matrix::zeros_generic(ns_virtual_dim, ns_virtual_dim);
        Q_terminal[(ns, ns)] = 200.0;

        // Input difference penalties
        // TODO: Support a configurable penalty multiplier
        let mut R: Vector<M::NI> = nalgebra::zero();
        R[0] = 30.0;
        R[1] = 50.0;
        let R = Matrix::from_diagonal(&R);

        let mut ineq_sparsity = MatrixMN::from_element_generic(Dy::new(2), ns_virtual_dim, false);
        ineq_sparsity[(0, 0)] = true;
        ineq_sparsity[(0, 1)] = true;
        ineq_sparsity[(0, ns)] = true;
        ineq_sparsity[(1, ns)] = true;

        MpcDistance {
            base: MpcBase::new(model, N, Q, Q_terminal, R, &ineq_sparsity),
            track: track.clone(),
        }
    }

    fn name() -> &'static str {
        "mpc_distance"
    }

    fn update_input_bounds(&mut self, u_min: Vector<M::NI>, u_max: Vector<M::NI>) {
        self.base.update_input_bounds(u_min, u_max)
    }

    fn step(
        &mut self,
        model: &M,
        dt: float,
        x: &Vector<M::NS>,
        u: &Matrix<M::NI, Dy>,
        p: &Vector<M::NP>,
        time_limit: Duration,
    ) -> (&Matrix<M::NI, Dy>, &Matrix<M::NS, Dy>) {
        let mut s_prev = self.track
            .centreline_distance(x[0], x[1])
            .expect("car outside track bounds");

        let track = &self.track;
        let base = &mut self.base;
        base.step(model, dt, x, u, p, time_limit, |_i, x_i, _u_i, stage| {
            // Find track point
            let guard = flame::start_guard("centreline point lookup");
            let (s, inside_track) = if let Some(s) = track.centreline_distance(x_i[0], x_i[1]) {
                (s, true)
            } else {
                (s_prev, false)
            };
            s_prev = s;
            let centreline_point = track.nearest_centreline_point(s);
            guard.end();

            let a_i = centreline_point.a(x_i[0], x_i[1]);
            let J = centreline_point.jacobian(a_i);

            // Only maxmisie distance for horizon points inside the track.
            if inside_track {
                // Minimise an approximate time penalty
                // Increasing speed is more important when minimising time if the car is travelling
                // slowly than if it is travelling fast.
                // TODO: Support a configurable penalty multiplier
                // TODO: Use a proper linearisation with a speed dependence
                // Ensure we don't divide by zero
                let time_penalty = -J.row(0).transpose() / max(x_i[3], 0.01);
                stage
                    .x_linear_penalty
                    .fixed_rows_mut::<U2>(0)
                    .copy_from(&time_penalty);
            }

            // Constraint the first virtual parameter to be a.
            stage
                .stage_ineq
                .fixed_slice_mut::<U1, U2>(0, 0)
                .copy_from(&J.row(1));
            stage.stage_ineq[(0, M::NS::dim())] = -1.0;
            stage.stage_ineq_min[0] = -a_i;
            stage.stage_ineq_max[0] = -a_i;

            // Constrain the car to remain within track bounds.
            // TODO: Make car width a configurable parameter
            let max_car_dimension = 0.22;
            let a_max = (centreline_point.track_width - max_car_dimension) / 2.0;
            stage.stage_ineq[(1, M::NS::dim())] = 1.0;
            stage.stage_ineq_min[1] = -a_max;
            stage.stage_ineq_max[1] = a_max;
        })
    }
}
