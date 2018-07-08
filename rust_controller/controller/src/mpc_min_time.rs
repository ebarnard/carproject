use flame;
use nalgebra::{MatrixMN, U1, U2};
use std::sync::Arc;
use std::time::Duration;

use control_model::ControlModel;
use prelude::*;
use track::TrackAndLookup;
use {Controller, InitController, MpcBase};

#[derive(Deserialize)]
pub struct Config {
    pub Q_terminal_v: float,
    pub R: Vec<float>,
}

pub struct MpcMinTime<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    base: MpcBase<M>,
    track: Arc<TrackAndLookup>,
}

impl<M: ControlModel> InitController<M> for MpcMinTime<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    type Config = Config;

    fn new(model: &M, N: u32, track: &Arc<TrackAndLookup>, config: Config) -> MpcMinTime<M> {
        let ns = M::NS::dim();
        let ns_dim = Dy::new(ns);

        // State penalties
        let Q = Matrix::zeros_generic(ns_dim, ns_dim);

        let mut Q_terminal = Matrix::zeros_generic(ns_dim, ns_dim);
        Q_terminal[(3, 3)] = config.Q_terminal_v;

        // Input difference penalties
        let R = Matrix::from_diagonal(&Vector::<M::NI>::from_row_slice(&config.R));

        let mut ineq_sparsity = MatrixMN::from_element_generic(Dy::new(1), ns_dim, false);
        ineq_sparsity[(0, 0)] = true;
        ineq_sparsity[(0, 1)] = true;

        MpcMinTime {
            base: MpcBase::new(model, N, Q, Q_terminal, R, &ineq_sparsity),
            track: track.clone(),
        }
    }

    fn name() -> &'static str {
        "mpc_min_time"
    }
}

impl<M: ControlModel> Controller<M> for MpcMinTime<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
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

            // Only maximise distance for horizon points inside the track.
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
        })
    }
}
