use std::sync::Arc;
use ui;
use ui::colors::*;
use ui::plot::{Axes, AxesRange, AxesScale, Line, SingleLineAxes};

use prelude::*;
use control_model::{Control, State as ControllerState};
use track::Track;

pub use ui::{EventSender, Window};

pub fn new() -> (ui::Window, ui::EventSender<Event>) {
    let vis = Visualisation {
        history: Vec::new(),
        track_inner: Line::new(0, BLACK),
        track_outer: Line::new(0, BLACK),
    };

    ui::Window::new(vis)
}

struct Visualisation {
    history: Vec<History>,
    track_inner: Line,
    track_outer: Line,
}

impl ui::State for Visualisation {
    type Event = Event;

    fn update(&mut self, event: Event) {
        match event {
            Event::Reset {
                n_history,
                track,
                horizon_len,
                np,
                n_cars,
            } => {
                let (track_inner, track_outer) = track_inner_outer(&track, 1000);
                self.track_inner = track_inner;
                self.track_outer = track_outer;
                self.history = vec![History::new(n_history, horizon_len, np); n_cars as usize];
            }
            Event::Record(car_idx, record) => self.history[car_idx as usize].record(record),
        }
    }

    fn draw(&self, c: &mut ui::Canvas) {
        let (w, h) = c.size();

        // 2/3 height for track
        c.subview([0.0, 0.0, w, h * 2.0 / 3.0], &mut |c| {
            ui::pad_all(10.0, c, |c| {
                let axes = Axes::new(
                    AxesRange::GrowShrink,
                    AxesRange::GrowShrink,
                    AxesScale::Same,
                );
                let mut track_lines = Vec::with_capacity(2 + self.history.len() * 2);
                track_lines.push(&self.track_inner);
                track_lines.push(&self.track_outer);
                for history in &self.history {
                    track_lines.push(&history.position);
                    track_lines.push(&history.predicted_horizon);
                }
                axes.draw(&track_lines, c);
            });
        });

        // 1/3 for everything else
        let n_cars = self.history.len();
        for (i, history) in self.history.iter().enumerate() {
            let subview_h = h / (3.0 * n_cars as f64);
            let subview_top = h * 2.0 / 3.0 + i as f64 * subview_h;

            c.subview([0.0, subview_top, w, subview_h], &mut |c| {
                let (w, h) = c.size();
                let plot_w = w / 4.0;

                c.subview([0.0 * plot_w, 0.0, plot_w, h], &mut |c| {
                    ui::pad_all(10.0, c, |c| history.v.draw(c));
                });

                c.subview([1.0 * plot_w, 0.0, plot_w, h], &mut |c| {
                    ui::pad_all(10.0, c, |c| history.heading.draw(c));
                });

                c.subview([2.0 * plot_w, 0.0, plot_w, h], &mut |c| {
                    ui::pad_all(10.0, c, |c| history.throttle_position.draw(c));
                });

                c.subview([3.0 * plot_w, 0.0, plot_w, h], &mut |c| {
                    ui::pad_all(10.0, c, |c| history.steering_angle.draw(c));
                });
            });
        }
    }
}

fn track_inner_outer(track: &Track, n: usize) -> (Line, Line) {
    let total_s = track.total_distance();
    let mut track_inner = Line::new(n, BLUE);
    let mut track_outer = Line::new(n, BLUE);

    for i in 0..n {
        let s = total_s * (i as float) / (n as float - 1.0);
        let point = track.nearest_centreline_point(s);
        let half_dx = -point.dy_ds * point.track_width * 0.5;
        let half_dy = point.dx_ds * point.track_width * 0.5;
        track_inner.push_back(point.x - half_dx, point.y - half_dy);
        track_outer.push_back(point.x + half_dx, point.y + half_dy);
    }

    (track_inner, track_outer)
}

#[derive(Clone)]
struct History {
    horizon_len: usize,
    np: usize,
    position: Line,
    heading: SingleLineAxes,
    v: SingleLineAxes,
    throttle_position: SingleLineAxes,
    steering_angle: SingleLineAxes,
    params: Vec<SingleLineAxes>,
    predicted_horizon: Line,
}

pub enum Event {
    Reset {
        n_history: usize,
        track: Arc<Track>,
        horizon_len: usize,
        np: usize,
        n_cars: u32,
    },
    Record(u32, Record),
}

pub struct Record {
    pub t: float,
    pub predicted_state: ControllerState,
    pub control: Control,
    pub params: Vec<float>,
    pub param_var: Vec<float>,
    pub predicted_horizon: Vec<(float, float)>,
}

impl History {
    pub fn new(n: usize, horizon_len: usize, np: usize) -> History {
        History {
            horizon_len,
            np,
            position: Line::new(n, BLACK),
            heading: SingleLineAxes::new(
                AxesRange::GrowShrink,
                AxesRange::Grow,
                AxesScale::Different,
                n,
                BLACK,
            ),
            v: SingleLineAxes::new(
                AxesRange::GrowShrink,
                AxesRange::Grow,
                AxesScale::Different,
                n,
                BLACK,
            ),
            throttle_position: SingleLineAxes::new(
                AxesRange::GrowShrink,
                AxesRange::Grow,
                AxesScale::Different,
                n,
                BLACK,
            ),
            steering_angle: SingleLineAxes::new(
                AxesRange::GrowShrink,
                AxesRange::Grow,
                AxesScale::Different,
                n,
                BLACK,
            ),
            params: (0..np)
                .map(|_| {
                    SingleLineAxes::new(
                        AxesRange::GrowShrink,
                        AxesRange::Grow,
                        AxesScale::Different,
                        n,
                        BLACK,
                    )
                })
                .collect(),
            predicted_horizon: Line::new(horizon_len + 1, RED),
        }
    }

    fn record(&mut self, r: Record) {
        assert_eq!(r.params.len(), r.param_var.len());
        assert_eq!(r.params.len(), self.np);
        assert_eq!(r.predicted_horizon.len(), self.horizon_len);

        let (x, y) = r.predicted_state.position;
        self.position.push_back(x, y);
        self.heading.push_back(r.t, r.predicted_state.heading);
        let v = float::hypot(r.predicted_state.velocity.0, r.predicted_state.velocity.1);
        self.v.push_back(r.t, v);
        self.throttle_position
            .push_back(r.t, r.control.throttle_position);
        self.steering_angle.push_back(r.t, r.control.steering_angle);

        for (line, value) in self.params.iter_mut().zip(r.params) {
            line.push_back(r.t, value);
        }

        // Start at the end of the current track and continue throughout the horizon
        self.predicted_horizon.push_back(x, y);
        for (x, y) in r.predicted_horizon {
            self.predicted_horizon.push_back(x, y);
        }
    }
}
