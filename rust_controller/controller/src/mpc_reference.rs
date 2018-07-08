use flame;
use log::Level::Debug;
use nalgebra::{MatrixMN, U1, U2, U3};
use std::sync::Arc;
use std::time::Duration;

use control_model::ControlModel;
use prelude::*;
use track::TrackAndLookup;
use {Controller, InitController, MpcBase};

#[derive(Deserialize)]
pub struct Config {
    pub Q: Vec<float>,
    pub R: Vec<float>,
    pub v_target: float,
}

pub struct MpcReference<M: ControlModel>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    base: MpcBase<M>,
    track: Arc<TrackAndLookup>,
}

impl<M: ControlModel> InitController<M> for MpcReference<M>
where
    DefaultAllocator: ModelDims<M::NS, M::NI, M::NP>,
{
    type Config = Config;

    fn new(model: &M, N: u32, track: &Arc<TrackAndLookup>, config: Config) -> MpcReference<M> {
        let ns_dim = Dy::new(M::NS::dim());

        // State penalties
        let mut Q = Vector::zeros_generic(ns_dim, U1);
        Q.fixed_rows_mut::<U3>(0)
            .copy_from(&Vector::<U3>::from_row_slice(&config.Q));
        let Q = Matrix::from_diagonal(&Q);

        let Q_terminal = Q.clone();

        // Input difference penalties
        let R = Matrix::from_diagonal(&Vector::<M::NI>::from_row_slice(&config.R));

        let mut ineq_sparsity = MatrixMN::from_element_generic(Dy::new(1), ns_dim, false);
        ineq_sparsity[(0, 0)] = true;
        ineq_sparsity[(0, 1)] = true;

        MpcReference {
            base: MpcBase::new(model, N, Q, Q_terminal, R, &ineq_sparsity),
            track: track.clone(),
        }
    }

    fn name() -> &'static str {
        "mpc_reference"
    }
}

impl<M: ControlModel> Controller<M> for MpcReference<M>
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
        let v_target = dt * 0.5;
        let mut s_target = flame::span_of("centreline distance lookup", || {
            self.track
                .centreline_distance(x[0], x[1])
                .expect("car outside track bounds")
        });

        let track = &self.track;
        let base = &mut self.base;
        base.step(model, dt, x, u, p, time_limit, |_i, x_i, _u_i, stage| {
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
            stage.x_target[0] = target.x;
            stage.x_target[1] = target.y;
            stage.x_target[2] = theta;

            // Track bounds inequality.
            if let Some(s) = track.centreline_distance(x_i[0], x_i[1]) {
                let centreline_point = track.nearest_centreline_point(s);
                let a_i = centreline_point.a(x_i[0], x_i[1]);
                let J = centreline_point.jacobian(a_i);
                let max_car_dimension = 0.2;
                let a_max = (centreline_point.track_width - max_car_dimension) / 2.0;

                stage
                    .stage_ineq
                    .fixed_slice_mut::<U1, U2>(0, 0)
                    .copy_from(&J.row(1));
                stage.stage_ineq_min[0] = -a_max - a_i;
                stage.stage_ineq_max[0] = a_max - a_i;
            } else {
                // Relax the constraint if the horizon point is outside the track.
                stage.stage_ineq_min[0] = NEG_INFINITY;
                stage.stage_ineq_max[0] = INFINITY;
            }
        })
    }
}
