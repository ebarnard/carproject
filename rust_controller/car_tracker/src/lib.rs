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
        background_sub_and_thresh(frame, &self.bg, 50, &mut self.fg_thresh);

        // Fill `fg` with the coordinates of foreground pixels.
        // The image is scaled to look at only one in every four pixels.
        find_foreground_positions(
            &self.fg_thresh,
            self.x_bins.len() as u32,
            None,
            None,
            4,
            &mut self.fg,
        );

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
        let x_min = x_median_scaled.saturating_sub(search_radius / 2);
        let y_min = y_median_scaled.saturating_sub(search_radius / 2);
        find_foreground_positions(
            &self.fg_thresh,
            self.x_bins.len() as u32,
            Some((x_min, x_min + search_radius)),
            Some((y_min, y_min + search_radius)),
            1,
            &mut self.fg,
        );

        // Find the median foreground pixel coordinate of the restricted image.
        let (x_median, y_median) = find_median_x_y(&self.fg, &mut self.x_bins, &mut self.y_bins);

        // Find the mean angle between the median coordinate and the other coordinates.
        let mut theta_vector = find_theta(&self.fg, x_median, y_median);

        // Prevent discontinuities in theta by assuming it never changes by more than 180deg in a
        // single frame.
        if theta_vector.dot(&self.theta_vector_prev) < 0.0 {
            theta_vector *= -1.0;
        }
        let theta = f64::atan2(theta_vector[1], theta_vector[0]);
        self.theta_vector_prev = theta_vector;

        (x_median as float, y_median as float, theta)
    }
}

fn background_sub_and_thresh(frame: &[u8], bg: &[u8], thresh: u8, fg_thresh: &mut [u8]) {
    assert_eq!(frame.len(), bg.len());
    assert_eq!(frame.len(), fg_thresh.len());

    // This simpler code optimises very poorly so we use explicit SIMD.
    //for ((&a, &b), out) in frame.iter().zip(&self.bg).zip(&mut self.fg_thresh) {
    //    *out = a <= b || a - b <= thresh
    //    if  {
    //        *out = 0;
    //    } else {
    //        *out = 1;
    //    }
    //}
    (frame.simd_iter(u8s(0)), bg.simd_iter(u8s(0)))
        .zip()
        .simd_map(|(a, b)| a.saturating_sub(b).gt(u8s(thresh)).as_u8s())
        .scalar_fill(fg_thresh);
}

fn find_foreground_positions(
    fg_thresh: &[u8],
    width: u32,
    x_range: Option<(u32, u32)>,
    y_range: Option<(u32, u32)>,
    scale_step: u32,
    fg: &mut Vec<(u32, u32)>,
) {
    let width = width as usize;
    let height = fg_thresh.len() / width;
    assert_eq!(width * height, fg_thresh.len());

    let (x_min, x_max) = x_range
        .map(|(l, u)| (l as usize, min(u as usize, width)))
        .unwrap_or((0, width));
    let (y_min, y_max) = y_range
        .map(|(l, u)| (l as usize, min(u as usize, height)))
        .unwrap_or((0, height));
    assert!(x_min <= x_max);
    assert!(y_min <= y_max);

    fg.truncate(0);

    // Exclude pixels outside y_range
    let fg_thresh_y_range = &fg_thresh[y_min * width..y_max * width];
    for (y, row) in fg_thresh_y_range
        .chunks(width)
        .enumerate()
        .step_by(scale_step as usize)
    {
        // Exclude pixels outside x_range
        let row_x_range = &row[x_min..x_max];
        for (x, _) in row_x_range
            .iter()
            .enumerate()
            .step_by(scale_step as usize)
            .filter(|&(_, &v)| v != 0)
        {
            fg.push(((x_min + x) as u32, (y_min + y) as u32));
        }
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

fn find_theta(fg: &[(u32, u32)], x_median: u32, y_median: u32) -> Vector2<float> {
    let mut position_cov = Matrix2::zeros();

    for &(x, y) in fg {
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
    let position_cov = position_cov / fg.len() as f64;

    // Find largest eigenvalue and corresponding eigenvector of the covariance matrix.
    let eigen = position_cov.symmetric_eigen();
    let max_eigenvalue_idx = eigen.eigenvalues.iamax();
    let theta_vector = eigen.eigenvectors.column(max_eigenvalue_idx);
    theta_vector.into_owned()
}
