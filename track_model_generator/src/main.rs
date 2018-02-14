extern crate piston_window;

use piston_window::*;
use std::f64;
use std::f64::consts::PI;

mod office_desk_track;
use office_desk_track as track;

const PI_2: f64 = 2.0 * PI;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const YELLOW: [f32; 4] = [0.4, 0.4, 1.0, 1.0];
const MAGENTA: [f32; 4] = [0.9, 0.9, 0.0, 1.0];

#[derive(Debug)]
enum Segment {
    Arc {
        x: f64,
        y: f64,
        start: f64,
        end: f64,
        r: f64,
        d: Direction,
    },
    Line { x1: f64, y1: f64, x2: f64, y2: f64 },
}

impl Segment {
    pub fn length(&self) -> f64 {
        match self {
            &Segment::Arc { r, start, end, d, .. } => {
                let (start, end) = match d {
                    Direction::Anticlockwise => (start, end),
                    Direction::Clockwise => (end, start),
                };
                r * norm(end - start)
            }
            &Segment::Line { x1, y1, x2, y2 } => f64::hypot(x2 - x1, y2 - y1),
        }
    }
}

fn norm(mut angle: f64) -> f64 {
    while angle < 0.0 {
        angle += PI_2;
    }
    while angle >= PI_2 {
        angle -= PI_2;
    }
    angle
}

fn tangents(points: &[CircleS]) -> Vec<Segment> {
    // Pair up circles 1-2, 2-3, 3-4, 4-1
    let pairs = Iterator::zip(points.iter(), points.iter().cycle().skip(1));

    let arcs = pairs.enumerate().map(|(i, (first, second))| {
        println!("Calculating pair {}", i);

        // Non centerline crossing case - always two or infinite tangents
        // direction always preserved
        let X = second.x - first.x;
        let Y = second.y - first.y;
        let R = second.r - first.r;
        let center_dist = f64::hypot(X, Y);

        // Same for both variants
        let mut atan = f64::atan(Y / X);

        // Correct for angles and shit - quadrants 2 and 3
        if X < 0.0 {
            atan = atan + PI;
        }

        let acos = f64::acos(-R / center_dist);

        let theta = acos + atan;

        // Not validly placed circles
        assert_eq!(theta.is_nan(), false);

        // First non-overlapping tangent
        let theta_d = direction(X, Y, theta);
        if theta_d == first.d && theta_d == second.d {
            println!("Type 1");
            verify(theta, X, Y, R);
            return (theta, theta, true);
        }

        // Other non-overlapping tangent
        let theta_d = theta_d.swap();
        if theta_d == first.d && theta_d == second.d {
            println!("Type 2");
            let theta = PI_2 - acos + atan;
            verify(theta, X, Y, R);
            return (theta, theta, true);
        }

        // Centerline crossing case - two if non-overlapping, one if touching, zero if crossing
        // direction always flipped
        let R = -second.r - first.r;
        let d = center_dist + R;

        let epsilon = 0.1;

        if d < -epsilon {
            panic!("no more cases");
        }

        let acos = f64::acos(-R / center_dist);

        let theta = acos + atan;

        // Touching case - no joining line
        if d.abs() < epsilon {
            println!("Type 3");
            verify(theta, X, Y, R);
            return (theta, theta + PI, false);
        }

        let theta_d = direction(X, Y, theta);

        // First overlapping case
        if theta_d == first.d && theta_d.swap() == second.d {
            println!("Type 4");
            verify(theta, X, Y, R);
            return (theta, theta + PI, true);
        }

        // Second overlapping case
        if d != 0.0 && theta_d.swap() == first.d && theta_d == second.d {
            println!("Type 5");
            let theta = PI_2 - acos + atan;
            verify(theta, X, Y, R);
            return (theta, theta + PI, true);
        }

        panic!("no more cases");
    });

    let mut segments = Vec::with_capacity(points.len() * 2);
    let mut next_arc_start = -1.0;

    for (i, (start, end, needs_line)) in arcs.enumerate() {
        let start_point = &points[i];
        let end_point = &points[(i + 1) % points.len()];

        let start = norm(start);
        let end = norm(end);

        segments.push(Segment::Arc {
            x: start_point.x,
            y: start_point.y,
            r: start_point.r,
            start: next_arc_start,
            end: start,
            d: start_point.d,
        });

        next_arc_start = end;

        if needs_line {
            let (sin_theta_start, cos_theta_start) = start.sin_cos();
            let (sin_theta_end, cos_theta_end) = end.sin_cos();
            segments.push(Segment::Line {
                x1: start_point.x + start_point.r * cos_theta_start,
                y1: start_point.y + start_point.r * sin_theta_start,
                x2: end_point.x + end_point.r * cos_theta_end,
                y2: end_point.y + end_point.r * sin_theta_end,
            })
        }
    }

    if let Segment::Arc { ref mut start, .. } = segments[0] {
        *start = next_arc_start;
    } else {
        unreachable!();
    }

    segments
}

fn direction(center_x: f64, center_y: f64, theta: f64) -> Direction {
    println!("direction({}, {}, {})", center_x, center_y, theta);
    // Find direction - using force like moment arm, sign of vector from center to pos cross
    let (normal_y, normal_x) = f64::sin_cos(theta);
    let cross = normal_x * center_y - center_x * normal_y;

    assert!(cross != 0.0);

    if cross > 0.0 {
        Direction::Anticlockwise
    } else {
        Direction::Clockwise
    }
}

fn verify(theta: f64, x: f64, y: f64, r: f64) {
    println!("verify(theta: {}, x: {}, y: {}, r: {})", theta, x, y, r);
    let (sin, cos) = theta.sin_cos();
    let res = x * cos + y * sin + r;
    if res.abs() > 0.01 {
        println!("verify() failed. res: {}", res);
    }
}

fn ex_points(n: u32, segments: &[Segment], width: f64) -> Vec<(f64, f64, f64, f64)> {
    let point_distance: f64 = segments.iter().map(Segment::length).sum::<f64>() / (n as f64);

    let mut segments = segments.iter();
    let mut current = segments.next().unwrap();
    let mut start = 0.0;

    let mut prev_dr = (1.0, 1.0);

    (0..n)
        .map(|i| {
            let target = (i as f64) * point_distance;

            let mut current_length = current.length();

            while start + current_length <= target {
                start += current_length;
                current = segments.next().unwrap();
                current_length = current.length();
            }

            let segment_s = target - start;

            match current {
                &Segment::Arc { r, d, x, y, start, .. } => {
                    let theta_add = segment_s / r;
                    let theta = norm(match d {
                        Direction::Anticlockwise => start + theta_add,
                        Direction::Clockwise => start - theta_add,
                    });

                    let (sin_theta, cos_theta) = theta.sin_cos();

                    let x = x + r * cos_theta;
                    let y = y + r * sin_theta;
                    let dx = width * cos_theta;
                    let dy = width * sin_theta;

                    (x, y, dx, dy)
                }
                &Segment::Line { x1, y1, x2, y2, .. } => {
                    let l = current_length;
                    let cos_theta = (x2 - x1) / l;
                    let sin_theta = (y2 - y1) / l;

                    let x = x1 + segment_s * cos_theta;
                    let y = y1 + segment_s * sin_theta;

                    let dx = -width * sin_theta;
                    let dy = width * cos_theta;

                    (x, y, dx, dy)
                }
            }
        })
        .map(|(x, y, dx, dy)| {
            // Ensure dr keeps pointing inwards or outwards
            // dr_i dot dr_i-1 will be positive if theta between them is small
            // therefore if the dot product is negative we reverse the direction of dr
            let (dx, dy) = if dx * prev_dr.0 + dy * prev_dr.1 < 0.0 {
                (-dx, -dy)
            } else {
                (dx, dy)
            };
            prev_dr = (dx, dy);
            (x, y, dx, dy)
        })
        .collect()
}

fn main() {
    let points = track::DEF;
    let track_width = track::WIDTH;

    let segments = tangents(points);

    println!("{:?}", segments);
    println!("{}", segments.iter().map(Segment::length).sum::<f64>());

    // Save output
    let pts = ex_points(2500, &segments, track_width);
    let mut f = ::std::fs::File::create("track2500.csv")
        .unwrap();
    for (a, b, c, d) in pts {
        use std::io::Write;
        let w = f64::hypot(c, d);
        writeln!(f, "{:.7},{:.7},{:.7}", a, b, w).unwrap();
    }
    drop(f);

    let scalaing_factor = 300.0;

    let x_min = points.iter().map(|p| p.x - p.r).fold(0. / 0., f64::min) - track_width;
    let y_min = points.iter().map(|p| p.y - p.r).fold(0. / 0., f64::min) - track_width;
    let x_max = points.iter().map(|p| p.x + p.r).fold(0. / 0., f64::max) + track_width;
    let y_max = points.iter().map(|p| p.y + p.r).fold(0. / 0., f64::max) + track_width;

    let x_shift = -x_min;
    let y_shift = -y_min;
    let x_size = x_max - x_min;
    let y_size = y_max - y_min;

    println!("x_shift: {}, y_shift: {}, x_size: {}, y_size: {}",
             x_shift,
             y_shift,
             x_size,
             y_size);

    let mut window: PistonWindow = WindowSettings::new("Hello World!",
                                                       [(x_size * scalaing_factor).ceil() as u32,
                                                        (y_size * scalaing_factor).ceil() as u32])
        .build()
        .unwrap();

    let e = window.next().unwrap();

    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g| {
            let c = c.scale(scalaing_factor, -scalaing_factor).trans(x_shift, y_shift - y_size);

            clear([1.0, 1.0, 1.0, 1.0], g);

            for p in points {
                let x = p.x - p.r;
                let y = p.y - p.r;
                let w = p.r * 2.0;
                let t = track_width * 0.5;

                circle_arc(RED,
                           t,
                           0.0,
                           PI * 1.99999,
                           [x, y, w, w], // rectangle
                           c.transform,
                           g);
            }

            for s in &segments {
                if let &Segment::Arc { x, y, start, end, r, d } = s {
                    let x = x - r;
                    let y = y - r;
                    let w = r * 2.0;
                    let t = track_width * 0.5;

                    let (start, end) = match d {
                        Direction::Anticlockwise => (start, end),
                        Direction::Clockwise => (end, start),
                    };

                    circle_arc(MAGENTA,
                               t,
                               start,
                               end,
                               [x, y, w, w], // rectangle
                               c.transform,
                               g);
                }

            }

            for s in &segments {
                if let &Segment::Line { x1, y1, x2, y2 } = s {
                    line(YELLOW, track_width * 0.5, [x1, y1, x2, y2], c.transform, g);
                }
            }

        });
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    Clockwise,
    Anticlockwise,
}

impl Direction {
    fn swap(self) -> Direction {
        match self {
            Direction::Clockwise => Direction::Anticlockwise,
            Direction::Anticlockwise => Direction::Clockwise,
        }
    }
}

pub struct CircleS {
    x: f64,
    y: f64,
    r: f64,
    d: Direction,
}
