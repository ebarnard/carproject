use csv;
use kdtree::kdtree::{Kdtree, KdtreePointTrait};
use nalgebra::{Matrix2, U2, LU};
use std::path::Path;

use prelude::*;
use cubic_spline::CubicSpline;

pub struct Track {
    pub x: Vec<float>,
    pub y: Vec<float>,
    pub dx: Vec<float>,
    pub dy: Vec<float>,
}

impl Track {
    pub fn load<P: AsRef<Path>>(path: P) -> Track {
        let mut track = Track {
            x: Vec::new(),
            y: Vec::new(),
            dx: Vec::new(),
            dy: Vec::new(),
        };
        let mut reader = csv::Reader::from_path(path).expect("Unable to open track");
        for row in reader.deserialize() {
            let (x, y, dx, dy): (float, float, float, float) = row.expect("Error reading track");
            track.x.push(x);
            track.y.push(y);
            track.dx.push(dx);
            track.dy.push(dy);
            debug!("row {} {}", x, y);
        }
        track
    }
}

pub struct Centreline {
    total_distance: float,
    /// Distance to the end of each spline segment
    cumulative_distance: Vec<float>,
    x_spline: CubicSpline,
    y_spline: CubicSpline,
    widths: Vec<float>,
    dt_ds: float,
}

#[derive(Clone, Copy, Debug)]
pub struct CentrelinePoint {
    pub x: float,
    pub y: float,
    pub dx_ds: float,
    pub dy_ds: float,
    pub dx_ds2: float,
    pub dy_ds2: float,
    pub track_width: float,
}

impl Centreline {
    /// Creates a spline track model from a given track.
    ///
    /// Panics if track points are not equally spaced.
    pub fn from_track(track: &Track) -> Centreline {
        let n = track.x.len();

        let x_spline = CubicSpline::periodic(&track.x);
        let y_spline = CubicSpline::periodic(&track.y);
        let widths = track
            .dx
            .iter()
            .zip(&track.dy)
            .map(|(&dx, &dy)| float::hypot(dx, dy))
            .collect();

        let x0 = x_spline.evaluate(0.0).0;
        let y0 = y_spline.evaluate(0.0).0;

        let ds_dt: Vec<_> = (0..n)
            .map(|i| (i + 1) as float)
            .map(|t| (x_spline.evaluate(t).0, y_spline.evaluate(t).0))
            .scan((x0, y0), |acc, (x, y)| {
                let ds = float::hypot(x - acc.0, y - acc.1);
                *acc = (x, y);
                Some(ds)
            })
            .collect();

        let cumulative_distance: Vec<_> = ds_dt
            .iter()
            .scan(0.0, |s, ds| {
                *s += ds;
                Some(*s)
            })
            .collect();

        let total_distance = cumulative_distance[n - 1];

        let centreline = Centreline {
            total_distance,
            cumulative_distance,
            x_spline,
            y_spline,
            widths,
            dt_ds: n as float / ds_dt.iter().sum::<float>(),
        };

        // TODO: Combine this check with ds_dt calculation.
        for &s in &centreline.cumulative_distance {
            let p = centreline.nearest_point(s);
            let d = float::hypot(p.dx_ds, p.dy_ds);
            assert!(
                (d - 1.0).abs() < 1e05,
                "track points are not equally spaced"
            );
        }

        centreline
    }

    pub fn nearest_point(&self, s: float) -> CentrelinePoint {
        let s = s % self.total_distance;
        let i = self.cumulative_distance
            .binary_search_by(|probe| probe.partial_cmp(&s).unwrap())
            .unwrap_or_else(|i| i);

        let s_segment_end = self.cumulative_distance[i];
        let s_segment_start = if i == 0 {
            0.0
        } else {
            self.cumulative_distance[i - 1]
        };

        let t = i as float + (s - s_segment_start) / (s_segment_end - s_segment_start);

        let (x, dx_dt, dx_dt2) = self.x_spline.evaluate(t);
        let (y, dy_dt, dy_dt2) = self.y_spline.evaluate(t);

        let point = CentrelinePoint {
            x,
            y,
            dx_ds: dx_dt * self.dt_ds,
            dy_ds: dy_dt * self.dt_ds,
            dx_ds2: dx_dt2 * self.dt_ds * self.dt_ds,
            dy_ds2: dy_dt2 * self.dt_ds * self.dt_ds,
            track_width: self.widths[i],
        };

        debug!("centreline point at s={} x={} y={}", s, point.x, point.y);

        point
    }
}

impl CentrelinePoint {
    /// Returns the perpendicular distance from the track centreline.
    /// Right-handed (90deg anticlockwise) positive.
    pub fn a(&self, x: float, y: float) -> float {
        // The sign on a is calculated using a dot product of the normal gradient and the
        // centreline error.
        let x_normal = x - self.x;
        let y_normal = y - self.y;
        let a_sign = (x_normal * -self.dy_ds + y_normal * self.dx_ds).signum();
        a_sign * float::hypot(x_normal, y_normal)
    }

    /// Returns the jacobian of the track parameterisation evalualted at (s, a).
    /// [delta_s, delta_a]' = J * [delta_x, delta_y]'
    pub fn jacobian(&self, a: float) -> Matrix<U2, U2> {
        let dx_ds = self.dx_ds;
        let dy_ds = self.dy_ds;
        let dx_ds2 = self.dx_ds2;
        let dy_ds2 = self.dy_ds2;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let J_inv = Matrix2::new(
            dx_ds - a * dy_ds2, -dy_ds,
            dy_ds + a * dx_ds2, dx_ds,
        );

        LU::new(J_inv)
            .try_inverse()
            .expect("jacobian must be invertible")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexedPoint(float, [f64; 2]);

impl KdtreePointTrait for IndexedPoint {
    fn dims(&self) -> &[f64] {
        &self.1[..]
    }
}

pub struct CentrelineLookup {
    kd: Kdtree<IndexedPoint>,
}

impl CentrelineLookup {
    pub fn from_centreline(centreline: &Centreline) -> CentrelineLookup {
        let mut points = Vec::new();
        for &s in &centreline.cumulative_distance {
            let point = centreline.nearest_point(s);
            let spade_point = IndexedPoint(s, [point.x as f64, point.y as f64]);
            points.push(spade_point);
        }

        let kd = Kdtree::new(&mut points);

        CentrelineLookup { kd }
    }

    pub fn centreline_distance(&self, x: float, y: float) -> float {
        let nn = self.kd
            .nearest_search(&IndexedPoint(0.0, [x as f64, y as f64]));
        debug!(
            "closest centreline point to ({}, {}) is ({}, {}) with s {}",
            x, y, nn.1[0], nn.1[1], nn.0
        );
        nn.0
    }
}
