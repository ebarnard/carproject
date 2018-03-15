#![allow(non_snake_case)]

extern crate csv;
extern crate cubic_spline;
#[macro_use]
extern crate log;
extern crate prelude;

use cubic_spline::CubicSpline;
use nalgebra::{Matrix2, Matrix3, Vector3};
use std::cell::Cell;
use std::path::Path;

use prelude::*;

pub struct Track {
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

impl Track {
    pub fn load<P: AsRef<Path>>(path: P) -> Track {
        let (mut xs, mut ys, mut widths) = (Vec::new(), Vec::new(), Vec::new());
        let mut reader = csv::Reader::from_path(path).expect("unable to open track");
        for row in reader.deserialize() {
            let (x, y, width): (float, float, float) = row.expect("error reading track");
            xs.push(x);
            ys.push(y);
            widths.push(width);
        }
        Track::from_track(&xs, &ys, &widths)
    }

    /// Creates a spline track model from a given track.
    ///
    /// Panics if track points are not equally spaced.
    pub fn from_track(x: &[float], y: &[float], width: &[float]) -> Track {
        let n = x.len();
        assert_eq!(n, y.len());
        assert_eq!(n, width.len());

        let cumulative_distance: Vec<_> = (1..n + 1)
            .scan((0.0, x[0], y[0]), |acc, t| {
                let t = t % n;
                let (x, y) = (x[t], y[t]);
                let s = acc.0 + float::hypot(x - acc.1, y - acc.2);
                *acc = (s, x, y);
                Some(s)
            })
            .collect();

        let total_distance = cumulative_distance[n - 1];
        let dt_ds = n as float / total_distance;

        let x_spline = CubicSpline::periodic(x);
        let y_spline = CubicSpline::periodic(y);

        // Check points are equally spaced
        for t in 0..n {
            let t = t as float;
            let (dx_dt, dy_dt) = (x_spline.evaluate(t).1, y_spline.evaluate(t).1);
            let ds = float::hypot(dx_dt * dt_ds, dy_dt * dt_ds);
            assert!(
                (ds - 1.0).abs() < 1e05,
                "track points are not equally spaced"
            );
        }

        Track {
            total_distance,
            cumulative_distance,
            x_spline,
            y_spline,
            widths: width.to_vec(),
            dt_ds,
        }
    }

    pub fn total_distance(&self) -> float {
        self.total_distance
    }

    pub fn nearest_centreline_point(&self, s: float) -> CentrelinePoint {
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

    /// Returns a bitmap mask of the track projected onto a camera frame.
    /// H is a homography mapping world into pixel coordinates.
    pub fn bitmap_mask(&self, H: &Matrix3<float>, width: u32, height: u32) -> Vec<u8> {
        let width = width as usize;
        let mut mask = vec![0; width * height as usize];

        let world_to_pixels = |(world_x, world_y)| {
            let world_pos = Vector3::new(world_x, world_y, 1.0);
            let image_pos = H * world_pos;
            let image_pos = image_pos / image_pos[2];
            (image_pos[0], image_pos[1])
        };

        self.visit_track_quads(|_, prev_outer, prev_inner, outer, inner| {
            let prev_outer = world_to_pixels(prev_outer);
            let prev_inner = world_to_pixels(prev_inner);
            let outer = world_to_pixels(outer);
            let inner = world_to_pixels(inner);

            draw_tri(&mut mask, width, outer, inner, prev_inner, 255);
            draw_tri(&mut mask, width, outer, prev_outer, prev_inner, 255);
        });

        mask
    }

    fn visit_track_quads<F>(&self, mut f: F)
    where
        F: FnMut(usize, (float, float), (float, float), (float, float), (float, float)),
    {
        let point = self.nearest_centreline_point(0.0);
        let mut prev_outer = point.at_a(point.track_width / 2.0);
        let mut prev_inner = point.at_a(-point.track_width / 2.0);

        for (i, &s) in self.cumulative_distance.iter().enumerate() {
            let point = self.nearest_centreline_point(s);
            let outer = point.at_a(point.track_width / 2.0);
            let inner = point.at_a(-point.track_width / 2.0);

            f(i, prev_outer, prev_inner, outer, inner);

            prev_outer = outer;
            prev_inner = inner;
        }
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

    /// Returns the track coordinate at a perpendicular distance `a` from the centreline.
    pub fn at_a(&self, a: float) -> (float, float) {
        let dx = self.dx_ds;
        let dy = self.dy_ds;
        let len = float::hypot(dx, dy);
        let dx = dx / len;
        let dy = dy / len;

        let x = self.x - dy * a;
        let y = self.y + dx * a;
        (x, y)
    }

    /// Returns the jacobian of the track parameterisation evalualted at (s, a).
    /// [delta_s, delta_a]' = J * [delta_x, delta_y]'
    pub fn jacobian(&self, a: float) -> Matrix2<float> {
        let dx_ds = self.dx_ds;
        let dy_ds = self.dy_ds;
        let dx_ds2 = self.dx_ds2;
        let dy_ds2 = self.dy_ds2;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let J_inv = Matrix2::new(
            dx_ds - a * dy_ds2, -dy_ds,
            dy_ds + a * dx_ds2, dx_ds,
        );

        J_inv.try_inverse().expect("jacobian must be invertible")
    }
}

pub struct Lookup {
    image: Vec<u16>,
    width: usize,
    height: usize,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
}

impl Lookup {
    pub fn new(track: &Track, width: usize, height: usize) -> Lookup {
        // Find track bounds
        let x_min = Cell::new(INFINITY);
        let x_max = Cell::new(NEG_INFINITY);
        let y_min = Cell::new(INFINITY);
        let y_max = Cell::new(NEG_INFINITY);
        let min_max = |(x, y)| {
            x_min.set(min(x_min.get(), x));
            x_max.set(max(x_max.get(), x));
            y_min.set(min(y_min.get(), y));
            y_max.set(max(y_max.get(), y));
        };
        for &s in &track.cumulative_distance {
            let p = track.nearest_centreline_point(s);
            min_max(p.at_a(p.track_width / 2.0));
            min_max(p.at_a(-p.track_width / 2.0));
        }

        let mut lookup = Lookup {
            image: vec![u16::max_value(); width * height],
            width,
            height,
            x_min: x_min.get(),
            x_max: x_max.get(),
            y_min: y_min.get(),
            y_max: y_max.get(),
        };

        track.visit_track_quads(|i, prev_outer, prev_inner, outer, inner| {
            let i = i as u16;
            let prev_outer = lookup.world_to_pixels(prev_outer);
            let prev_inner = lookup.world_to_pixels(prev_inner);
            let outer = lookup.world_to_pixels(outer);
            let inner = lookup.world_to_pixels(inner);
            draw_tri(&mut lookup.image, width, outer, inner, prev_inner, i);
            draw_tri(&mut lookup.image, width, outer, prev_outer, prev_inner, i);
        });

        lookup
    }

    pub fn centreline_distance(&self, track: &Track, x: float, y: float) -> Option<float> {
        if x < self.x_min || x > self.x_max || y < self.y_min || y > self.y_max {
            return None;
        }
        let (x_pixel, y_pixel) = self.world_to_pixels((x, y));

        let i = self.image[y_pixel as usize * self.width + x_pixel as usize];
        if i == u16::max_value() {
            return None;
        }

        let s = track.cumulative_distance[i as usize];
        debug!("track distance at ({}, {}) is {}", x, y, s);
        Some(s)
    }

    fn world_to_pixels(&self, (x, y): (float, float)) -> (float, float) {
        let x_pixel = (self.width - 1) as float * (x - self.x_min) / (self.x_max - self.x_min);
        let y_pixel = (self.height - 1) as float * (y - self.y_min) / (self.y_max - self.y_min);
        (x_pixel, y_pixel)
    }
}

pub struct TrackAndLookup {
    pub track: Track,
    pub lookup: Lookup,
}

impl TrackAndLookup {
    pub fn new(track: Track, lookup_width: usize, lookup_height: usize) -> TrackAndLookup {
        let lookup = Lookup::new(&track, lookup_width, lookup_height);
        TrackAndLookup { track, lookup }
    }
}

impl TrackAndLookup {
    pub fn nearest_centreline_point(&self, s: float) -> CentrelinePoint {
        self.track.nearest_centreline_point(s)
    }

    pub fn centreline_distance(&self, x: float, y: float) -> Option<float> {
        self.lookup.centreline_distance(&self.track, x, y)
    }
}

// A wild triangle rasterisation algorithm appears. This probably isn't the best place for it.
// It is based on the triangle drawing algorithm from:
// http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
fn draw_tri<T: Copy>(
    image: &mut [T],
    width: usize,
    v1: (float, float),
    v2: (float, float),
    v3: (float, float),
    colour: T,
) {
    assert!(image.len() % width == 0);
    let height = image.len() / width;

    // Sort vertices by ascending y then x coordinate.
    let mut vertices = [v1, v2, v3];
    vertices.sort_by(|&(a1, a2), &(b1, b2)| (a2, a1).partial_cmp(&(b2, b1)).unwrap());

    // Axes are positive rightwards and upwards.
    // Bottom left triagle
    let v1 = vertices[0];
    // Middle, bottom right, or upper left triangle.
    let v2 = vertices[1];
    // Upper right triangle.
    let v3 = vertices[2];

    let y_bottom = v1.1 as isize;
    // The is the y scanline at which the gradient of one of the triangle sides changes.
    let y_mid = v2.1 as isize;
    let y_top = v3.1 as isize;

    // The left side of the triangle is nominally the one with vertex part way up it.
    let mut dx_dy_left = (v2.0 - v1.0) / (v2.1 - v1.1);
    let dx_dy_right = (v3.0 - v1.0) / (v3.1 - v1.1);

    let mut x_left = v1.0;
    let mut x_right = v1.0;

    // Don't attempt to draw beyond the top of the image.
    for scan_y in y_bottom..min(y_top, height as isize) {
        // At y_mid change the left side gradient and left x position.
        if scan_y == y_mid {
            dx_dy_left = (v3.0 - v2.0) / (v3.1 - v2.1);
            x_left = v2.0;
        }

        // Don't draw parts of triangles below the bottom of the image.
        if scan_y >= 0 {
            let scan_x_left = min(min(x_left as usize, x_right as usize), width);
            let scan_x_right = min(max(x_left as usize, x_right as usize), width);
            for i in scan_x_left..scan_x_right {
                image[scan_y as usize * width + i] = colour;
            }
        }

        // Update x positions for next y scanline.
        x_left += dx_dy_left;
        x_right += dx_dy_right;
    }
}
