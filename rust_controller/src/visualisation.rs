use gnuplot::{AxesCommon, Color, Figure};

use prelude::*;
use controller::State as ControllerState;
use simulation_model::State;
use track::Track;

#[derive(Default)]
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
    v_predicted: Vec<float>,
    param_error: Vec<float>,
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
            v_predicted: Vec::with_capacity(n),
            param_error: Vec::with_capacity(n),
        }
    }

    pub fn record(
        &mut self,
        t: float,
        state: &State,
        controller_state: &ControllerState,
        param_err: float,
    ) {
        self.t.push(t);
        let (x, y) = state.position;
        let v = float::hypot(state.velocity.0, state.velocity.1);
        self.x.push(x);
        self.y.push(y);
        /*let heading = if self.heading.len() > 0 {
            phase_unwrap(self.heading[0], state.heading)
        } else {
            state.heading
        };*/
        self.heading.push(state.heading);
        self.v.push(v);
        let controller_v = float::hypot(controller_state.velocity.0, controller_state.velocity.1);
        self.x_error.push(controller_state.position.0 - x);
        self.y_error.push(controller_state.position.1 - y);
        self.heading_error.push(controller_state.heading - state.heading);
        self.v_error.push(controller_v - v);
        self.v_predicted.push(controller_v);
        self.param_error.push(param_err);
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
    // v predicted
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 7);
        ax.set_title("v predicted", &[]);
        ax.lines(&history.t, &history.v_predicted, &[]);
    }
    // param error
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(3, 3, 8);
        ax.set_title("param error", &[]);
        ax.lines(&history.t, &history.param_error, &[]);
    }
    fg.show();
}
