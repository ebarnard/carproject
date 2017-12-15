extern crate nalgebra;
#[cfg(target_os = "macos")]
extern crate nalgebra_lapack;

use nalgebra::{Dynamic, MatrixMN, U1};
#[cfg(target_os = "macos")]
use nalgebra_lapack::Cholesky;
#[cfg(not(target_os = "macos"))]
use nalgebra::Cholesky;

#[allow(non_camel_case_types)]
type float = f64;

pub struct CubicSpline {
    coefs: Vec<(float, float, float, float)>,
}

impl CubicSpline {
    pub fn periodic(y: &[float]) -> CubicSpline {
        // Maths from: http://mathworld.wolfram.com/CubicSpline.html

        let n = y.len();
        assert!(n >= 2);

        let mut mat = MatrixMN::from_diagonal_element_generic(Dynamic::new(n), Dynamic::new(n), 4.);
        for i in 0..n - 1 {
            mat[(i, i + 1)] = 1.;
            mat[(i + 1, i)] = 1.;
        }
        mat[(0, n - 1)] = 1.;
        mat[(n - 1, 0)] = 1.;

        let mut vec = MatrixMN::zeros_generic(Dynamic::new(n), U1);
        vec[0] = 3. * (y[1] - y[n - 1]);
        for i in 1..n - 1 {
            vec[i] = 3. * (y[i + 1] - y[i - 1]);
        }
        vec[n - 1] = 3. * (y[0] - y[n - 2]);

        let chol = Cholesky::new(mat).unwrap();
        chol.solve_mut(&mut vec);

        let make_coef = |y, y_p, D, D_p| {
            let a = y;
            let b = D;
            let c = 3.0 * (y_p - y) - 2.0 * D - D_p;
            let d = 2.0 * (y - y_p) + D + D_p;
            (a, b, c, d)
        };

        let mut coefs = Vec::with_capacity(n);
        for i in 0..n - 1 {
            coefs.push(make_coef(y[i], y[i + 1], vec[i], vec[i + 1]));
        }
        coefs.push(make_coef(y[n - 1], y[0], vec[n - 1], vec[0]));

        CubicSpline { coefs: coefs }
    }

    fn get_coefs(&self, t: float) -> (float, float, float, float) {
        let index = (t.floor() as usize) % self.coefs.len();
        self.coefs[index]
    }

    pub fn evaluate(&self, t: float) -> (float, float, float) {
        let (a, b, c, d) = self.get_coefs(t);
        let t = t.fract();
        let t_2 = t * t;
        let t_3 = t_2 * t;

        let x = a + b * t + c * t_2 + d * t_3;
        let x_dash = b + 2.0 * c * t + 3.0 * d * t_2;
        let x_dash_2 = 2.0 * c + 6.0 * d * t;

        (x, x_dash, x_dash_2)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matlab_d_values() {
        let y = &[5., 6., 8., 2., 1.5, 7.4, 9.];
        let D_expected = &[
            -2.509756097560976,
            3.519512195121950,
            -2.568292682926829,
            -5.246341463414635,
            4.053658536585368,
            5.231707317073170,
            -2.480487804878049,
        ];

        let spline = CubicSpline::periodic(y);

        for (D, &D_exp) in spline.coefs.iter().map(|&(_, D, _, _)| D).zip(D_expected) {
            assert!((D - D_exp).abs() < 1e-6, "{} {}", D_exp, D);
        }
    }
}
