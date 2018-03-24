use std::cell::Cell;
use std::collections::VecDeque;
use std::f64::{INFINITY, NEG_INFINITY};
use std::fmt::Write;

use {max, min, Canvas, Color, TextXAlign, TextYAlign};
use colors::BLACK;

#[derive(Clone, Copy)]
pub enum AxesRange {
    GrowShrink,
    Grow,
    Fixed(f64, f64),
}

#[derive(Clone, Copy)]
pub enum AxesScale {
    Same,
    Different,
}

#[derive(Clone)]
pub struct Axes {
    x_range: AxesRange,
    y_range: AxesRange,
    scale: AxesScale,
    x_min: Cell<f64>,
    x_max: Cell<f64>,
    y_min: Cell<f64>,
    y_max: Cell<f64>,
}

impl Axes {
    pub fn new(x_range: AxesRange, y_range: AxesRange, scale: AxesScale) -> Axes {
        Axes {
            x_range,
            y_range,
            scale,
            x_min: Cell::new(INFINITY),
            x_max: Cell::new(NEG_INFINITY),
            y_min: Cell::new(INFINITY),
            y_max: Cell::new(NEG_INFINITY),
        }
    }

    pub fn draw(&self, lines: &[&Line], canvas: &mut Canvas) {
        const AXIS_DEPTH: f64 = 20.0;
        const TICK_LENGTH: f64 = 5.0;
        const TICK_DENSITY: f64 = 0.02;

        fn min_max<F>(lines: &[&Line], f: F) -> (f64, f64)
        where
            F: Fn(&Line) -> (f64, f64),
        {
            lines
                .iter()
                .fold((INFINITY, NEG_INFINITY), |(acc_min, acc_max), l| {
                    let (l_min, l_max) = f(l);
                    (min(acc_min, l_min), max(acc_max, l_max))
                })
        }

        let (mut x_min, mut x_max) = match self.x_range {
            AxesRange::GrowShrink => min_max(lines, |l| (l.x_min, l.x_max)),
            AxesRange::Grow => {
                let (x_min, x_max) = min_max(lines, |l| (l.x_min, l.x_max));
                self.x_min.set(min(x_min, self.x_min.get()));
                self.x_max.set(max(x_max, self.x_max.get()));
                (self.x_min.get(), self.x_max.get())
            }
            AxesRange::Fixed(x, w) => (x, x + w),
        };

        let (mut y_min, mut y_max) = match self.y_range {
            AxesRange::GrowShrink => min_max(lines, |l| (l.y_min, l.y_max)),
            AxesRange::Grow => {
                let (y_min, y_max) = min_max(lines, |l| (l.y_min, l.y_max));
                self.y_min.set(min(y_min, self.y_min.get()));
                self.y_max.set(max(y_max, self.y_max.get()));
                (self.y_min.get(), self.y_max.get())
            }
            AxesRange::Fixed(y, h) => (y, y + h),
        };

        let (w, h) = canvas.size();

        let mut x_scale = (w - AXIS_DEPTH) / (x_max - x_min);
        let mut y_scale = (h - AXIS_DEPTH) / (y_max - y_min);

        if let AxesScale::Same = self.scale {
            // Keep proportions
            let scale = min(x_scale, y_scale);
            let x_offset = 0.5 * (x_scale - scale) / scale * (x_max - x_min);
            let y_offset = 0.5 * (y_scale - scale) / scale * (y_max - y_min);
            x_scale = scale;
            y_scale = scale;
            x_min -= x_offset;
            x_max += x_offset;
            y_min -= y_offset;
            y_max += y_offset;
        }

        let mut tick_str = String::new();

        // Plot x axis
        canvas.subview(
            [AXIS_DEPTH, h - AXIS_DEPTH, w - AXIS_DEPTH, AXIS_DEPTH],
            &mut |canvas| {
                let (w, _) = canvas.size();

                // Draw ticks
                let target_n_ticks = w * TICK_DENSITY;
                let mut ticks = Ticks::new(x_min, x_max, target_n_ticks);
                while let Some(tick) = ticks.next(&mut tick_str) {
                    let tick_x = x_scale * (tick - x_min);
                    canvas.line(BLACK, 0.5, [tick_x, 0.0, tick_x, TICK_LENGTH]);
                    canvas.text(
                        BLACK,
                        12,
                        TextXAlign::Centre,
                        TextYAlign::Bottom,
                        &tick_str,
                        [tick_x, AXIS_DEPTH],
                    );
                }

                // Draw axis
                canvas.line(BLACK, 0.5, [0.0, 0.0, w, 0.0]);
            },
        );
        // Plot y axis
        canvas.subview([0.0, 0.0, AXIS_DEPTH, h - AXIS_DEPTH], &mut |canvas| {
            let (_, h) = canvas.size();

            // Draw ticks
            let target_n_ticks = h * TICK_DENSITY;
            let mut ticks = Ticks::new(y_min, y_max, target_n_ticks);
            while let Some(tick) = ticks.next(&mut tick_str) {
                let tick_y = y_scale * (tick - y_min);
                canvas.line(
                    BLACK,
                    0.5,
                    [AXIS_DEPTH - TICK_LENGTH, h - tick_y, AXIS_DEPTH, h - tick_y],
                );
                canvas.text(
                    BLACK,
                    12,
                    TextXAlign::Right,
                    TextYAlign::Middle,
                    &tick_str,
                    [AXIS_DEPTH - TICK_LENGTH, h - tick_y],
                );
            }

            // Draw axis
            canvas.line(BLACK, 0.5, [AXIS_DEPTH, 0.0, AXIS_DEPTH, h]);
        });

        // Plot lines
        canvas.subview(
            [AXIS_DEPTH, 0.0, w - AXIS_DEPTH, h - AXIS_DEPTH],
            &mut |canvas| {
                let (_, h) = canvas.size();

                for lines in &*lines {
                    for (&(x1, y1), &(x2, y2)) in lines.vals.iter().zip(lines.vals.iter().skip(1)) {
                        canvas.line(
                            lines.color,
                            0.5,
                            [
                                x_scale * (x1 - x_min),
                                h - y_scale * (y1 - y_min),
                                x_scale * (x2 - x_min),
                                h - y_scale * (y2 - y_min),
                            ],
                        );
                    }
                }
            },
        );
    }
}

struct Ticks {
    spacing: f64,
    pos: f64,
    max: f64,
    str_precision: usize,
}

impl Ticks {
    fn new(min_value: f64, max_value: f64, target_n_ticks: f64) -> Ticks {
        // Accept 1, 2 and 5 * powers of 10
        // 1 * 10^n, 2 * 10^n, 5 * 10^n

        let range_per_tick = (max_value - min_value) / target_n_ticks;
        let tick_power_of_10 = range_per_tick.log10().floor();
        let range_per_power_10 = 10.0f64.powf(tick_power_of_10);

        // See whether 1, 2 or five better gives TICK_DENSITY
        let n_1_x_tick = ((max_value - min_value) / range_per_power_10).floor();
        let n_2_x_tick = ((max_value - min_value) / (2.0 * range_per_power_10)).floor();
        let n_5_x_tick = ((max_value - min_value) / (5.0 * range_per_power_10)).floor();

        let diff_1_x_tick = (target_n_ticks - n_1_x_tick).abs();
        let diff_2_x_tick = (target_n_ticks - n_2_x_tick).abs();
        let diff_5_x_tick = (target_n_ticks - n_5_x_tick).abs();

        let range_per_tick = if diff_1_x_tick < diff_2_x_tick && diff_1_x_tick < diff_5_x_tick {
            range_per_power_10
        } else if diff_2_x_tick < diff_5_x_tick {
            2.0 * range_per_power_10
        } else {
            5.0 * range_per_power_10
        };

        let first_tick = range_per_tick * (min_value / range_per_tick).ceil();
        let str_precision = max(0.0, -tick_power_of_10.floor()) as usize;

        Ticks {
            spacing: range_per_tick,
            pos: first_tick,
            max: max_value,
            str_precision,
        }
    }

    fn next(&mut self, tick_str: &mut String) -> Option<f64> {
        if self.spacing.is_nan() || self.spacing == 0.0 {
            return None;
        }

        if self.pos > self.max {
            return None;
        }

        let tick_val = self.pos;
        self.pos += self.spacing;

        tick_str.clear();
        let _ = write!(tick_str, "{:.*}", self.str_precision, tick_val);
        Some(tick_val)
    }
}

#[derive(Clone)]
pub struct Line {
    n: usize,
    vals: VecDeque<(f64, f64)>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    color: Color,
}

impl Line {
    pub fn new(n: usize, color: Color) -> Line {
        Line {
            n,
            vals: VecDeque::with_capacity(n),
            x_min: INFINITY,
            x_max: NEG_INFINITY,
            y_min: INFINITY,
            y_max: NEG_INFINITY,
            color,
        }
    }

    pub fn push_back(&mut self, x: f64, y: f64) {
        let mut x_min = self.x_min;
        let mut x_max = self.x_max;
        let mut y_min = self.y_min;
        let mut y_max = self.y_max;
        if self.vals.len() == self.n {
            let (front_x, front_y) = self.vals.pop_front().unwrap_or((NEG_INFINITY, INFINITY));
            if front_x <= x_min {
                x_min = self.vals.iter().fold(INFINITY, |acc, v| min(acc, v.0));
            }
            if front_x >= x_max {
                x_max = self.vals.iter().fold(NEG_INFINITY, |acc, v| max(acc, v.0));
            }
            if front_y <= y_min {
                y_min = self.vals.iter().fold(INFINITY, |acc, v| min(acc, v.1));
            }
            if front_y >= y_max {
                y_max = self.vals.iter().fold(NEG_INFINITY, |acc, v| max(acc, v.1));
            }
        }
        self.vals.push_back((x, y));
        self.x_min = min(x, x_min);
        self.x_max = max(x, x_max);
        self.y_min = min(y, y_min);
        self.y_max = max(y, y_max);
    }
}

#[derive(Clone)]
pub struct SingleLineAxes {
    pub axes: Axes,
    pub line: Line,
}

impl SingleLineAxes {
    pub fn new(
        x_range: AxesRange,
        y_range: AxesRange,
        scale: AxesScale,
        n: usize,
        color: Color,
    ) -> SingleLineAxes {
        SingleLineAxes {
            axes: Axes::new(x_range, y_range, scale),
            line: Line::new(n, color),
        }
    }

    pub fn push_back(&mut self, x: f64, y: f64) {
        self.line.push_back(x, y);
    }

    pub fn draw(&self, canvas: &mut Canvas) {
        self.axes.draw(&[&self.line], canvas);
    }
}
