use gnuplot::{AxesCommon, Color, Figure};
use std::collections::VecDeque;

use prelude::*;
use controller::{Control, State as ControllerState};
use simulation_model::State;
use track::Track;

pub struct History {
    n: usize,
    x: VecDeque<float>,
    y: VecDeque<float>,
    heading: VecDeque<float>,
    v: VecDeque<float>,
    t: VecDeque<float>,
    x_error: VecDeque<float>,
    y_error: VecDeque<float>,
    heading_error: VecDeque<float>,
    v_error: VecDeque<float>,
    throttle_position: VecDeque<float>,
    steering_angle: VecDeque<float>,
    np: Option<usize>,
    params: Vec<VecDeque<float>>,
    param_var: Vec<VecDeque<float>>,
}

impl History {
    pub fn new(n: usize) -> History {
        History {
            n,
            x: VecDeque::with_capacity(n),
            y: VecDeque::with_capacity(n),
            heading: VecDeque::with_capacity(n),
            v: VecDeque::with_capacity(n),
            t: VecDeque::with_capacity(n),
            x_error: VecDeque::with_capacity(n),
            y_error: VecDeque::with_capacity(n),
            heading_error: VecDeque::with_capacity(n),
            v_error: VecDeque::with_capacity(n),
            throttle_position: VecDeque::with_capacity(n),
            steering_angle: VecDeque::with_capacity(n),
            np: None,
            params: Vec::new(),
            param_var: Vec::new(),
        }
    }

    pub fn record<NP: DimName>(
        &mut self,
        t: float,
        state: &State,
        controller_state: &ControllerState,
        control: Control,
        params: &Vector<NP>,
        param_cov: &Matrix<NP, NP>,
    ) where
        DefaultAllocator: Dims2<NP, NP>,
    {
        let n = self.n;

        if let Some(np) = self.np {
            assert_eq!(NP::dim(), np);
        } else {
            let np = NP::dim();
            self.params = (0..np).map(|_| VecDeque::with_capacity(n)).collect();
            self.param_var = (0..np).map(|_| VecDeque::with_capacity(n)).collect();
            self.np = Some(np);
        }

        push_back_fixed_len(n, &mut self.t, t);
        let (x, y) = state.position;
        let v = float::hypot(state.velocity.0, state.velocity.1);
        push_back_fixed_len(n, &mut self.x, x);
        push_back_fixed_len(n, &mut self.y, y);
        push_back_fixed_len(n, &mut self.heading, state.heading);
        push_back_fixed_len(n, &mut self.v, v);
        push_back_fixed_len(n, &mut self.x_error, controller_state.position.0 - x);
        push_back_fixed_len(n, &mut self.y_error, controller_state.position.1 - y);
        push_back_fixed_len(
            n,
            &mut self.heading_error,
            controller_state.heading - state.heading,
        );
        let controller_v = float::hypot(controller_state.velocity.0, controller_state.velocity.1);
        push_back_fixed_len(n, &mut self.v_error, controller_v - v);
        push_back_fixed_len(n, &mut self.throttle_position, control.throttle_position);
        push_back_fixed_len(n, &mut self.steering_angle, control.steering_angle);
        extend_fixed_len(n, &mut self.params, params.as_slice());
        extend_fixed_len(n, &mut self.param_var, param_cov.diagonal().as_slice());
    }
}

fn push_back_fixed_len(n: usize, vec: &mut VecDeque<float>, val: float) {
    if vec.len() == n {
        vec.pop_front();
    }
    vec.push_back(val);
}

fn extend_fixed_len(n: usize, vecs: &mut Vec<VecDeque<float>>, vals: &[float]) {
    for (vec, &val) in vecs.iter_mut().zip(vals) {
        push_back_fixed_len(n, vec, val);
    }
}

pub fn plot(track: &Track, history: &History) {
    // Plot position
    let mut fg = Figure::new();
    fg.set_terminal("qt", "");
    {
        let n = 1000;
        let total_s = track.total_distance();
        let mut x_min = Vec::with_capacity(n);
        let mut y_min = Vec::with_capacity(n);
        let mut x_max = Vec::with_capacity(n);
        let mut y_max = Vec::with_capacity(n);

        for i in 0..n {
            let s = total_s * (i as float) / (n as float);
            let point = track.nearest_centreline_point(s);
            let half_dx = -point.dy_ds * point.track_width * 0.5;
            let half_dy = point.dx_ds * point.track_width * 0.5;
            x_min.push(point.x - half_dx);
            y_min.push(point.y - half_dy);
            x_max.push(point.x + half_dx);
            y_max.push(point.y + half_dy);
        }

        let ax = fg.axes2d();
        ax.lines(&x_min, &y_min, &[Color("green")]);
        ax.lines(&x_max, &y_max, &[Color("green")]);
        ax.lines(&history.x, &history.y, &[Color("black")]);
    }
    fg.show();

    // Plot state and state errors
    let nrows = 3;
    let ncols = 3;
    let mut i = 0;
    let mut fg = Figure::new();
    fg.set_terminal("qt", "");
    // Speed
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("v", &[]);
        ax.lines(&history.t, &history.v, &[]);
    }
    // Heading
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("heading", &[]);
        ax.lines(&history.t, &history.heading, &[]);
    }
    // X error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("x error", &[]);
        ax.lines(&history.t, &history.x_error, &[]);
    }
    // Y error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("y error", &[]);
        ax.lines(&history.t, &history.y_error, &[]);
    }
    // heading error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("heading error", &[]);
        ax.lines(&history.t, &history.heading_error, &[]);
    }
    // v error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("v error", &[]);
        ax.lines(&history.t, &history.v_error, &[]);
    }
    // throttle
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        i += 1;
        ax.set_title("throttle", &[]);
        ax.lines(&history.t, &history.throttle_position, &[]);
    }
    // steering
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        ax.set_title("steering", &[]);
        ax.lines(&history.t, &history.steering_angle, &[]);
    }
    fg.show();

    // Plot params and param standard deviations
    let mut fg = Figure::new();
    fg.set_terminal("qt", "");
    let np = history.np.unwrap() as u32;
    let ncols = float::from(np).sqrt().ceil() as u32;
    let nrows = 1 + (np - 1) / ncols;
    for (i, (param, var)) in history.params.iter().zip(&history.param_var).enumerate() {
        let i = i as u32;
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        ax.set_title(&format!("param {}", i), &[]);
        ax.lines(
            &history.t,
            param.iter().zip(var).map(|(p, v)| p + v.sqrt()),
            &[Color("red")],
        );
        ax.lines(
            &history.t,
            param.iter().zip(var).map(|(p, v)| p - v.sqrt()),
            &[Color("red")],
        );
        ax.lines(&history.t, param, &[]);
    }
    fg.show();
}
