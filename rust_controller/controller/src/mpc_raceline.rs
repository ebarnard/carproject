use flame;
use nalgebra::{self, MatrixMN, U1, U2};
use std::sync::Arc;
use std::time::Duration;

use control_model::ControlModel;
use prelude::*;
use track::{Raceline, TrackAndLookup};
use {Controller, MpcBase};

pub struct MpcRaceline<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    base: MpcBase<M>,
    track: Arc<TrackAndLookup>,
    raceline: Raceline,
}

impl<M: ControlModel> Controller<M> for MpcRaceline<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    fn new(model: &M, N: u32, track: &Arc<TrackAndLookup>) -> MpcRaceline<M> {
        let ns = M::NS::dim();
        let ns_virtual_dim = Dy::new(M::NS::dim() + 1);

        // State penalties
        let mut Q = Matrix::zeros_generic(ns_virtual_dim, ns_virtual_dim);
        Q[(ns, ns)] = 10.0;

        let Q_terminal = Q.clone();

        // Input difference penalties
        // TODO: Support a configurable penalty multiplier
        let mut R: Vector<M::NI> = nalgebra::zero();
        R[0] = 30.0;
        R[1] = 200.0;
        let R = Matrix::from_diagonal(&R);

        let mut ineq_sparsity = MatrixMN::from_element_generic(Dy::new(2), ns_virtual_dim, false);
        // track bounds constraint
        ineq_sparsity[(0, 0)] = true;
        ineq_sparsity[(0, 1)] = true;
        // a target virtual state
        ineq_sparsity[(1, 0)] = true;
        ineq_sparsity[(1, 1)] = true;
        ineq_sparsity[(1, ns)] = true;

        MpcRaceline {
            base: MpcBase::new(model, N, Q, Q_terminal, R, &ineq_sparsity),
            track: track.clone(),
            raceline: Raceline::load_for_track(&track.track),
        }
    }

    fn name() -> &'static str {
        "mpc_raceline"
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
        let mut s_prev = self
            .track
            .centreline_distance(x[0], x[1])
            .expect("car outside track bounds");

        let track = &self.track;
        let raceline = &self.raceline;
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

            // Constrain the car to remain within track bounds.
            // TODO: Make car width a configurable parameter
            let max_car_dimension = 0.1;
            let a_max = (centreline_point.track_width - max_car_dimension) / 2.0;
            stage
                .stage_ineq
                .fixed_slice_mut::<U1, U2>(0, 0)
                .copy_from(&J.row(1));
            stage.stage_ineq_min[0] = -a_i - a_max;
            stage.stage_ineq_max[0] = -a_i + a_max;

            // target a penalty
            // || a - a_target(s) ||
            // || a0 + δa - a_target(s0) - a_target' δs ||
            // || J_21 δx + J_22 δy - a_target' J_11 δx - a_target' J_12 δy + a0 - a_target(s0) ||
            let raceline_point = raceline.at(&centreline_point);
            stage.stage_ineq[(1, 0)] = J[(1, 0)] - J[(0, 0)] * raceline_point.da_ds;
            stage.stage_ineq[(1, 1)] = J[(1, 1)] - J[(0, 1)] * raceline_point.da_ds;
            stage.stage_ineq[(1, M::NS::dim())] = -1.0;
            stage.stage_ineq_min[1] = -a_i + raceline_point.a;
            stage.stage_ineq_max[1] = stage.stage_ineq_min[1];
        })
    }
}
