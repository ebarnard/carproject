use csv;
use kdtree::kdtree::{Kdtree, KdtreePointTrait};
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
}

#[derive(Clone, Copy, Debug)]
pub struct CentrelinePoint {
    pub x: float,
    pub y: float,
    pub dx_ds: float,
    pub dy_ds: float,
    pub track_width: float,
    pub kappa: float,
}

impl Centreline {
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

        let cumulative_distance: Vec<_> = (0..n)
            .map(|i| (i + 1) as float)
            .map(|s| (x_spline.evaluate(s).0, y_spline.evaluate(s).0))
            .scan((x0, y0), |acc, (x, y)| {
                let ds = float::hypot(x - acc.0, y - acc.1);
                *acc = (x, y);
                Some(ds)
            })
            .scan(0.0, |s, ds| {
                *s += ds;
                Some(*s)
            })
            .collect();

        let total_distance = cumulative_distance[n - 1];

        Centreline {
            total_distance,
            cumulative_distance,
            x_spline,
            y_spline,
            widths,
        }
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

        let (x, dx, dx2) = self.x_spline.evaluate(t);
        let (y, dy, dy2) = self.y_spline.evaluate(t);

        let kappa_num = dx * dy2 - dy * dx2;
        let kappa_denom = dx * dx + dy * dy;
        let kappa_denom = kappa_denom * kappa_denom.sqrt();
        let kappa = kappa_num / kappa_denom;

        let point = CentrelinePoint {
            x,
            y,
            // TODO: Fix _ds magnitude
            // dx_ds = dx_dt * dt_ds
            // dt_ds = 1 / spacing
            dx_ds: dx,
            dy_ds: dy,
            track_width: self.widths[i],
            kappa,
        };

        debug!("centreline point at s={} x={} y={}", s, point.x, point.y);

        point
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
