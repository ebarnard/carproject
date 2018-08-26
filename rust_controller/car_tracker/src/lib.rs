extern crate itertools;

extern crate prelude;

use itertools::Itertools;
use nalgebra::{Matrix2, Vector2};

use prelude::*;

mod aligned_buffer;
use aligned_buffer::AlignedBuffer;

mod camera;
pub use camera::{Camera, Capture};

const MAX_CARS: u32 = 2;
const SIMD_ALIGN: usize = 32;

pub struct Tracker {
    max_cars: u32,
    bg: AlignedBuffer,
    fg_thresh: AlignedBuffer,
    fg: Vec<(u32, u32)>,
    x_bins: Vec<u32>,
    y_bins: Vec<u32>,
    theta_vector_prev: [Vector2<float>; MAX_CARS as usize],
    car_positions: [Option<Car>; MAX_CARS as usize],
}

#[derive(Clone, Copy)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
}

impl Tracker {
    pub fn new(max_cars: u32, width: u32, height: u32, track_mask: &[u8], bg: &[u8]) -> Tracker {
        assert!(max_cars <= MAX_CARS);

        let pixels = width as usize * height as usize;
        assert_eq!(pixels, track_mask.len());
        assert_eq!(pixels, bg.len());

        // Everywhere the mask is zero is not track and should be excluded from the frame.
        // This is done by setting the background value to 255.
        let mut bg = AlignedBuffer::from_slice(&bg, SIMD_ALIGN);
        for (bg, &mask) in bg.iter_mut().zip(track_mask) {
            if mask == 0 {
                *bg = 255;
            }
        }

        Tracker {
            max_cars,
            bg,
            fg_thresh: AlignedBuffer::zeroed(pixels, SIMD_ALIGN),
            fg: vec![(0, 0); pixels],
            x_bins: vec![0; width as usize],
            y_bins: vec![0; height as usize],
            theta_vector_prev: [Vector2::zeros(); MAX_CARS as usize],
            car_positions: [None; MAX_CARS as usize],
        }
    }

    pub fn track_frame(&mut self, frame: &[u8]) -> &[Option<Car>] {
        assert_eq!(self.bg.len(), frame.len());

        // Subtract background and threshold at 50.
        background_sub_and_thresh(frame, &self.bg, 50, &mut self.fg_thresh);

        // Fill `fg` with the coordinates of foreground pixels.
        // The image is scaled to look at only one in every four pixels.
        find_foreground_positions(
            &self.fg_thresh,
            self.x_bins.len() as u32,
            None,
            |_| None,
            4,
            &mut self.fg,
        );

        // Don't bother looking for cars if there are no foreground pixels.
        if self.fg.len() == 0 {
            self.car_positions.iter_mut().for_each(|p| *p = None);
            return &self.car_positions[..self.max_cars as usize];
        }

        let (left, right) = match self.max_cars {
            1 => (self.track_frame_one_car(), None),
            2 => self.track_frame_two_cars(),
            _ => unreachable!(),
        };

        // The order of the cars can change so we assume the correct ordering is the one where
        // each car has moved the smallest distance.
        // `large_distance` is any number larger than a possible distance in the image.
        let large_distance = frame.len() as f64;
        let distance = |a: Option<Car>, b: Option<Car>| match (a, b) {
            (Some(a), Some(b)) => float::hypot(a.x - b.x, a.y - b.y),
            _ => large_distance,
        };

        let same_order_distance =
            distance(left, self.car_positions[0]) + distance(right, self.car_positions[1]);
        let swap_order_distance =
            distance(left, self.car_positions[1]) + distance(right, self.car_positions[0]);

        if same_order_distance <= swap_order_distance {
            self.car_positions[0] = left;
            self.car_positions[1] = right;
        } else {
            self.car_positions[0] = right;
            self.car_positions[1] = left;
        }

        &self.car_positions[..self.max_cars as usize]
    }

    fn track_frame_one_car(&mut self) -> Option<Car> {
        // Find the median foreground pixel coordinate of the scaled image.
        let (x_median_scaled, y_median_scaled) =
            find_median_x_y(&self.fg, |_, _| true, &mut self.x_bins, &mut self.y_bins);

        self.track_single_car(0, x_median_scaled, y_median_scaled)
    }

    fn track_frame_two_cars(&mut self) -> (Option<Car>, Option<Car>) {
        // Find mean pixel coordiante.
        let (x_mean, y_mean) = self
            .fg
            .iter()
            .fold((0, 0), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
        let (x_mean, y_mean) = (
            x_mean as f64 / self.fg.len() as f64,
            y_mean as f64 / self.fg.len() as f64,
        );

        // Find the direction of greatest variance. This is normal to the line dividing the cars.
        let divider_normal = find_theta(&self.fg, x_mean as u32, y_mean as u32);

        // Determine whether a pixel is to the "left" or "right" of the dividing line using its dot
        // product with the line normal vector. The direction of "left" and "right" is arbitrary.
        let left_of_divider = |x, y| {
            let delta_pos = Vector2::new(x as f64 - x_mean as f64, y as f64 - y_mean as f64);
            delta_pos.dot(&divider_normal) >= 0.0
        };

        // Find the median foreground pixel coordinate of the scaled image for the "left" car.
        let (left_x_median_scaled, left_y_median_scaled) = find_median_x_y(
            &self.fg,
            |x, y| left_of_divider(x, y),
            &mut self.x_bins,
            &mut self.y_bins,
        );

        // Find the median foreground pixel coordinate of the scaled image for the "right" car.
        let (right_x_median_scaled, right_y_median_scaled) = find_median_x_y(
            &self.fg,
            |x, y| !left_of_divider(x, y),
            &mut self.x_bins,
            &mut self.y_bins,
        );

        let left_pos = self.track_single_car(0, left_x_median_scaled, left_y_median_scaled);
        let right_pos = self.track_single_car(1, right_x_median_scaled, right_y_median_scaled);

        (left_pos, right_pos)
    }

    fn track_single_car(
        &mut self,
        i: usize,
        x_median_scaled: u32,
        y_median_scaled: u32,
    ) -> Option<Car> {
        // Fill `fg` with the coordinates of foreground pixels.
        // Only pixels around the scaled image median are considered.
        let search_width = 150;
        let x_min = x_median_scaled.saturating_sub(search_width / 2);
        let y_min = y_median_scaled.saturating_sub(search_width / 2);
        find_foreground_positions(
            &self.fg_thresh,
            self.x_bins.len() as u32,
            Some((y_min, y_min + search_width)),
            |_| Some((x_min, x_min + search_width)),
            1,
            &mut self.fg,
        );

        // Find the median foreground pixel coordinate of the restricted image.
        let (x_median, y_median) =
            find_median_x_y(&self.fg, |_, _| true, &mut self.x_bins, &mut self.y_bins);

        // Find the mean angle between the median coordinate and the other coordinates.
        let mut theta_vector = find_theta(&self.fg, x_median, y_median);

        // Prevent discontinuities in theta by assuming it never changes by more than 180deg in a
        // single frame.
        if theta_vector.dot(&self.theta_vector_prev[i]) < 0.0 {
            theta_vector *= -1.0;
        }
        let theta = f64::atan2(theta_vector[1], theta_vector[0]);
        self.theta_vector_prev[i] = theta_vector;

        // TODO: Return None if there is an unexpected number of foreground pixels.
        Some(Car {
            x: x_median as float,
            y: y_median as float,
            heading: theta,
        })
    }
}

fn background_sub_and_thresh(frame: &[u8], bg: &[u8], thresh: u8, fg_thresh: &mut [u8]) {
    assert_eq!(frame.len(), bg.len());
    assert_eq!(frame.len(), fg_thresh.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            return background_sub_and_thresh_avx2(frame, bg, thresh, fg_thresh);
        } else if is_x86_feature_detected!("sse2") {
            return background_sub_and_thresh_sse2(frame, bg, thresh, fg_thresh);
        }
    }

    background_sub_and_thresh_fallback(frame, bg, thresh, fg_thresh);
}

fn background_sub_and_thresh_fallback(frame: &[u8], bg: &[u8], thresh: u8, fg_thresh: &mut [u8]) {
    for ((&frame_pixel, &bg_pixel), out) in frame.iter().zip(bg).zip(fg_thresh) {
        if frame_pixel.checked_sub(bg_pixel).unwrap_or(0) > thresh {
            *out = 255;
        } else {
            *out = 0;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn background_sub_and_thresh_sse2(
    mut frame: &[u8],
    mut bg: &[u8],
    thresh: u8,
    fg_thresh: &mut [u8],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64 as x86;

    let thresh_vec = x86::_mm_set1_epi8(thresh as i8);

    let mut i = 0;
    while frame.len() >= 16 {
        let frame_vec = x86::_mm_loadu_si128(frame.as_ptr() as *const _);
        let bg_vec = x86::_mm_load_si128(bg.as_ptr() as *const _);
        let diff_vec = x86::_mm_subs_epu8(frame_vec, bg_vec);
        // SSE2 does not have unsigned comparison instructions so instead we perform another
        // saturating subtraction and exploit the fact that we only check that the resulting value
        // is not equal to zero.
        let fg_thresh_vec = x86::_mm_subs_epu8(diff_vec, thresh_vec);
        x86::_mm_store_si128(
            fg_thresh.as_mut_ptr().offset(i as isize) as *mut _,
            fg_thresh_vec,
        );

        frame = &frame[16..];
        bg = &bg[16..];
        i += 16;
    }

    // Process any remaining pixels if frame.len() is not a multiple of 16.
    background_sub_and_thresh_fallback(frame, bg, thresh, &mut fg_thresh[i..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn background_sub_and_thresh_avx2(
    mut frame: &[u8],
    mut bg: &[u8],
    thresh: u8,
    fg_thresh: &mut [u8],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64 as x86;

    let thresh_vec = x86::_mm256_set1_epi8(thresh as i8);

    let mut i = 0;
    while frame.len() >= 32 {
        let frame_vec = x86::_mm256_loadu_si256(frame.as_ptr() as *const _);
        let bg_vec = x86::_mm256_load_si256(bg.as_ptr() as *const _);
        let diff_vec = x86::_mm256_subs_epu8(frame_vec, bg_vec);
        // We use the same approach for the greater-than comparison as the SSE2 version.
        let fg_thresh_vec = x86::_mm256_subs_epu8(diff_vec, thresh_vec);
        x86::_mm256_store_si256(
            fg_thresh.as_mut_ptr().offset(i as isize) as *mut _,
            fg_thresh_vec,
        );

        frame = &frame[32..];
        bg = &bg[32..];
        i += 32;
    }

    // Process any remaining pixels if frame.len() is not a multiple of 32.
    background_sub_and_thresh_fallback(frame, bg, thresh, &mut fg_thresh[i..]);
}

fn find_foreground_positions<F: FnMut(u32) -> Option<(u32, u32)>>(
    fg_thresh: &[u8],
    width: u32,
    y_range: Option<(u32, u32)>,
    mut x_range: F,
    scale_step: u32,
    fg: &mut Vec<(u32, u32)>,
) {
    let width = width as usize;
    let height = fg_thresh.len() / width;
    assert_eq!(width * height, fg_thresh.len());

    let (y_min, y_max) = y_range
        .map(|(l, u)| (l as usize, min(u as usize, height)))
        .unwrap_or((0, height));
    assert!(y_min <= y_max);

    fg.truncate(0);

    // Exclude pixels outside y_range
    let fg_thresh_y_range = &fg_thresh[y_min * width..y_max * width];
    for (y, row) in fg_thresh_y_range
        .chunks(width)
        .enumerate()
        .step(scale_step as usize)
    {
        let (x_min, x_max) = x_range((y_min + y) as u32)
            .map(|(l, u)| (l as usize, min(u as usize, width)))
            .unwrap_or((0, width));
        assert!(x_min <= x_max);

        // Exclude pixels outside x_range
        let row_x_range = &row[x_min..x_max];
        for (x, _) in row_x_range
            .iter()
            .enumerate()
            .step(scale_step as usize)
            .filter(|&(_, &v)| v != 0)
        {
            fg.push(((x_min + x) as u32, (y_min + y) as u32));
        }
    }
}

fn find_median_x_y<F: FnMut(u32, u32) -> bool>(
    fg: &[(u32, u32)],
    mut include_pixel: F,
    x_bins: &mut [u32],
    y_bins: &mut [u32],
) -> (u32, u32) {
    x_bins.iter_mut().for_each(|v| *v = 0);
    y_bins.iter_mut().for_each(|v| *v = 0);

    let mut n_inliers = 0;
    for &(x, y) in fg {
        if include_pixel(x, y) {
            x_bins[x as usize] += 1;
            y_bins[y as usize] += 1;
            n_inliers += 1;
        }
    }

    let (x_median, _) = x_bins
        .iter()
        .scan(0, |acc, &n| {
            *acc += n;
            Some(*acc)
        }).enumerate()
        .filter(|&(_, acc)| acc >= n_inliers / 2)
        .next()
        .unwrap_or((0, 0));

    let (y_median, _) = y_bins
        .iter()
        .scan(0, |acc, &n| {
            *acc += n;
            Some(*acc)
        }).enumerate()
        .filter(|&(_, acc)| acc >= n_inliers / 2)
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
