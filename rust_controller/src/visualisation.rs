use gnuplot::{AxesCommon, Color, Figure};
use itertools::Itertools;

use prelude::*;
use controller::{Control, State as ControllerState};
use simulation_model::State;
use track::Track;

pub struct History {
    x: Vec<float>,
    y: Vec<float>,
    heading: Vec<float>,
    v: Vec<float>,
    t: Vec<float>,
    x_error: Vec<float>,
    y_error: Vec<float>,
    heading_error: Vec<float>,
    v_error: Vec<float>,
    control: Vec<Control>,
    np: Option<usize>,
    params: Vec<float>,
    param_var: Vec<float>,
}

impl History {
    pub fn new(n: usize) -> History {
        History {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            heading: Vec::with_capacity(n),
            v: Vec::with_capacity(n),
            t: Vec::with_capacity(n),
            x_error: Vec::with_capacity(n),
            y_error: Vec::with_capacity(n),
            heading_error: Vec::with_capacity(n),
            v_error: Vec::with_capacity(n),
            control: Vec::with_capacity(n),
            np: None,
            params: Vec::with_capacity(n),
            param_var: Vec::with_capacity(n),
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
        assert_eq!(NP::dim(), *self.np.get_or_insert_with(|| NP::dim()));

        self.t.push(t);
        let (x, y) = state.position;
        let v = float::hypot(state.velocity.0, state.velocity.1);
        self.x.push(x);
        self.y.push(y);
        self.heading.push(state.heading);
        self.v.push(v);
        let controller_v = float::hypot(controller_state.velocity.0, controller_state.velocity.1);
        self.x_error.push(controller_state.position.0 - x);
        self.y_error.push(controller_state.position.1 - y);
        self.heading_error.push(controller_state.heading - state.heading);
        self.v_error.push(controller_v - v);
        self.control.push(control);
        self.params.extend_from_slice(params.as_slice());
        self.param_var.extend_from_slice(param_cov.diagonal().as_slice());
    }
}

pub fn plot(track: &Track, history: &History) {
    let mut fg = Figure::new();
    fg.set_terminal("qt", "");
    // Position
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 0);
        ax.lines(&track.x, &track.y, &[Color("green")]);
        ax.lines(&history.x, &history.y, &[Color("black")]);
    }
    // Speed
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 1);
        ax.set_title("v", &[]);
        ax.lines(&history.t, &history.v, &[]);
    }
    // Heading
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 2);
        ax.set_title("heading", &[]);
        ax.lines(&history.t, &history.heading, &[]);
    }
    // X error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 3);
        ax.set_title("x error", &[]);
        ax.lines(&history.t, &history.x_error, &[]);
    }
    // Y error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 4);
        ax.set_title("y error", &[]);
        ax.lines(&history.t, &history.y_error, &[]);
    }
    // heading error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 5);
        ax.set_title("heading error", &[]);
        ax.lines(&history.t, &history.heading_error, &[]);
    }
    // v error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 6);
        ax.set_title("v error", &[]);
        ax.lines(&history.t, &history.v_error, &[]);
    }
    // throttle
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 7);
        ax.set_title("throttle", &[]);
        ax.lines(
            &history.t,
            history.control.iter().map(|c| c.throttle_position),
            &[],
        );
    }
    // steering
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 8);
        ax.set_title("steering", &[]);
        ax.lines(
            &history.t,
            history.control.iter().map(|c| c.steering_angle),
            &[],
        );
    }
    fg.show();

    // Plot params and param covariances
    let mut fg = Figure::new();
    fg.set_terminal("qt", "");
    let np = history.np.unwrap() as u32;
    let ncols = (np as float).sqrt().ceil() as u32;
    let nrows = 1 + (np - 1) / ncols;
    for i in 0..np {
        let ax = fg.axes2d();
        ax.set_pos_grid(nrows, ncols, i);
        ax.set_title(&format!("param {}", i), &[]);
        let param = history.params.iter().skip(i as usize).step(np as usize);
        let var = history.param_var.iter().skip(i as usize).step(np as usize);
        ax.lines(&history.t, param.clone().zip(var.clone()).map(|(p, v)| p + v), &[Color("red")]);
        ax.lines(&history.t, param.clone().zip(var).map(|(p, v)| p - v), &[Color("red")]);
        ax.lines(&history.t, param, &[]);
    }
    fg.show();
}
