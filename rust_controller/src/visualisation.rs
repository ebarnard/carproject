use prelude::*;
use simulation_model::State;
use track::Track;

#[derive(Default)]
pub struct History {
    x: Vec<float>,
    y: Vec<float>,
    heading: Vec<float>,
    v: Vec<float>,
    t: Vec<float>,
}

impl History {
    pub fn new(n: usize) -> History {
        History {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            heading: Vec::with_capacity(n),
            v: Vec::with_capacity(n),
            t: Vec::with_capacity(n),
        }
    }

    pub fn record(&mut self, t: float, state: &State) {
        self.t.push(t);
        let (x, y) = state.position;
        self.x.push(x);
        self.y.push(y);
        self.heading.push(state.heading);
        self.v.push(float::hypot(state.velocity.0, state.velocity.1));
    }
}

use gnuplot::{AxesCommon, Caption, Color, Figure};

pub fn plot(track: &Track, history: &History) {
    let mut fg = Figure::new();
    fg.set_terminal("png", "output.png");
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(2, 2, 0);
        ax.lines(&track.x, &track.y, &[Caption("A line"), Color("green")]);
        ax.lines(&history.x, &history.y, &[Caption("A line"), Color("black")]);
    }
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(2, 2, 1);
        ax.lines(&history.t, &history.v, &[]);
    }
    {
        let ax = fg.axes2d();
        ax.set_pos_grid(2, 2, 2);
        ax.lines(&history.t, &history.heading, &[]);
    }
    fg.show();
}
