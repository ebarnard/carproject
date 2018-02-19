#![feature(iterator_step_by)]

extern crate faster;
extern crate prelude;

use faster::*;
use nalgebra::{Matrix2, Vector2};
use prelude::*;

mod camera;
pub use camera::{Camera, Capture};

pub struct Tracker {
    bg: Vec<u8>,
    fg_thresh: Vec<u8>,
    fg: Vec<(u32, u32)>,
    x_bins: Vec<u32>,
    y_bins: Vec<u32>,
    theta_vector_prev: Vector2<float>,
}

impl Tracker {
    pub fn new(width: u32, height: u32, track_mask: &[u8], bg: &[u8]) -> Tracker {
        let pixels = width as usize * height as usize;
        assert_eq!(pixels, track_mask.len());
        assert_eq!(pixels, bg.len());

        // Everywhere the mask is zero is not track and should be excluded from the frame.
        // This is done by setting the background value to 255.
        let mut bg = bg.to_vec();
        for (bg, &mask) in bg.iter_mut().zip(track_mask) {
            if mask == 0 {
                *bg = 255;
            }
        }

        Tracker {
            bg,
            fg_thresh: vec![0; pixels],
            fg: vec![(0, 0); pixels],
            x_bins: vec![0; width as usize],
            y_bins: vec![0; height as usize],
            theta_vector_prev: Vector2::zeros(),
        }
    }

    pub fn track_frame(&mut self, frame: &[u8]) -> (float, float, float) {
        assert_eq!(self.bg.len(), frame.len());

        // Subtract background and threshold at 50.
        //
        // This simpler code optimises very poorly so we use explicit SIMD.
        //for ((&a, &b), out) in frame.iter().zip(&self.bg).zip(&mut self.fg_thresh) {
        //    *out = a <= b || a - b <= thresh
        //    if  {
        //        *out = 0;
        //    } else {
        //        *out = 1;
        //    }
        //}
        let thresh = 50;
        (frame.simd_iter(u8s(0)), self.bg.simd_iter(u8s(0)))
            .zip()
            .simd_map(|(a, b)| a.saturating_sub(b).gt(u8s(thresh)).as_u8s())
            .scalar_fill(&mut self.fg_thresh);

        // Fill `fg` with the coordinates of foreground pixels.
        // The image is scaled to look at only one in every four pixels.
        let scale_step = 4;
        self.fg.truncate(0);
        for (y, row) in self.fg_thresh
            .chunks(self.x_bins.len())
            .enumerate()
            .step_by(scale_step)
        {
            for (x, _) in row.iter()
                .enumerate()
                .step_by(scale_step)
                .filter(|&(_, &v)| v != 0)
            {
                self.fg.push((x as u32, y as u32));
            }
        }

        // Return early if there are no foreground pixels.
        // TODO: Return None if there is an unexpected number of foreground pixels.
        if self.fg.len() == 0 {
            return (0.0, 0.0, 0.0);
        }

        // Find the median foreground pixel coordinate of the scaled image
        let (x_median_scaled, y_median_scaled) =
            find_median_x_y(&self.fg, &mut self.x_bins, &mut self.y_bins);

        // Fill `fg` with the coordinates of foreground pixels.
        // Only pixels around the scaled image median are considered.
        let search_radius = 200;
        let x_min = x_median_scaled.saturating_sub(search_radius / 2) as usize;
        let y_min = y_median_scaled.saturating_sub(search_radius / 2) as usize;
        self.fg.truncate(0);
        for (y, row) in self.fg_thresh
            .chunks(self.x_bins.len())
            .enumerate()
            .skip(y_min)
            .take(search_radius as usize)
        {
            for (x, _) in row.iter()
                .enumerate()
                .skip(x_min)
                .take(search_radius as usize)
                .filter(|&(_, &v)| v != 0)
            {
                self.fg.push((x as u32, y as u32));
            }
        }

        // Find the median foreground pixel coordinate of the restricted image.
        let (x_median, y_median) = find_median_x_y(&self.fg, &mut self.x_bins, &mut self.y_bins);

        // Find the mean angle between the median coordinate and the other coordinates.
        let mut position_cov = Matrix2::zeros();
        for &(x, y) in &self.fg {
            let delta_x = x as f64 - x_median as f64;
            let delta_y = y as f64 - y_median as f64;
            // Normalise to reduce the influence of outliers
            // This is somewhat equivalent to finding the mean angle
            // Add eps to avoid dividing zero by zero
            let delta_norm = f64::hypot(delta_x, delta_y) + ::std::f64::EPSILON;
            let val = Vector2::new(delta_x / delta_norm, delta_y / delta_norm);
            position_cov += &val * val.transpose();
        }

        // Divide by the population size to make it a covariance matrix.
        let position_cov = position_cov / self.fg.len() as f64;

        // Find largest eigenvalue and corresponding eigenvector of the covariance matrix.
        let eigen = position_cov.symmetric_eigen();
        let max_eigenvalue_idx = eigen.eigenvalues.iamax();
        let theta_vector = eigen.eigenvectors.column(max_eigenvalue_idx);

        // Prevent discontinuities in theta by assuming it never changes by more than 180deg in a
        // single frame.
        let theta_vector = if theta_vector.dot(&self.theta_vector_prev) >= 0.0 {
            theta_vector.into_owned()
        } else {
            -theta_vector
        };
        let theta = f64::atan2(theta_vector[1], theta_vector[0]);
        self.theta_vector_prev = theta_vector;

        (x_median as float, y_median as float, theta)
    }
}

fn find_median_x_y(fg: &[(u32, u32)], x_bins: &mut [u32], y_bins: &mut [u32]) -> (u32, u32) {
    x_bins.iter_mut().for_each(|v| *v = 0);
    y_bins.iter_mut().for_each(|v| *v = 0);

    for &(x, y) in fg {
        x_bins[x as usize] += 1;
        y_bins[y as usize] += 1;
    }

    let mid = (fg.len() / 2) as u32;
    let (x_median, _) = x_bins
        .iter()
        .scan(0, |acc, &n| {
            *acc += n;
            Some(*acc)
        })
        .enumerate()
        .filter(|&(_, acc)| acc >= mid)
        .next()
        .unwrap_or((0, 0));

    let (y_median, _) = y_bins
        .iter()
        .scan(0, |acc, &n| {
            *acc += n;
            Some(*acc)
        })
        .enumerate()
        .filter(|&(_, acc)| acc >= mid)
        .next()
        .unwrap_or((0, 0));

    (x_median as u32, y_median as u32)
}
