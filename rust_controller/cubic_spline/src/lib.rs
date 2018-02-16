#![allow(non_snake_case)]

extern crate prelude;

use prelude::*;

#[derive(Clone)]
pub struct CubicSpline {
    coefs: Vec<(float, float, float, float)>,
}

impl CubicSpline {
    pub fn periodic(y: &[float]) -> CubicSpline {
        // Maths from: http://mathworld.wolfram.com/CubicSpline.html

        let n = y.len();
        assert!(n >= 2);

        let tri_a = vec![1.0; n];
        let tri_b = vec![4.0; n];
        let tri_c = vec![1.0; n];

        let mut tri_d = vec![0.0; n];
        tri_d[0] = 3. * (y[1] - y[n - 1]);
        for i in 1..n - 1 {
            tri_d[i] = 3. * (y[i + 1] - y[i - 1]);
        }
        tri_d[n - 1] = 3. * (y[0] - y[n - 2]);

        let tri_x = thomas_sherman_morrison(&tri_a, &tri_b, &tri_c, &tri_d);

        let make_coef = |y, y_p, D, D_p| {
            let a = y;
            let b = D;
            let c = 3.0 * (y_p - y) - 2.0 * D - D_p;
            let d = 2.0 * (y - y_p) + D + D_p;
            (a, b, c, d)
        };

        let mut coefs = Vec::with_capacity(n);
        for i in 0..n - 1 {
            coefs.push(make_coef(y[i], y[i + 1], tri_x[i], tri_x[i + 1]));
        }
        coefs.push(make_coef(y[n - 1], y[0], tri_x[n - 1], tri_x[0]));

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

// `a` is the subdiagonal, `b` is the main diagonal, `c` is the superdiagonal, and `d` is the
// right-hand side.
fn thomas_sherman_morrison(a: &[float], b: &[float], c: &[float], d: &[float]) -> Vec<float> {
    let N = a.len();
    assert_eq!(N, b.len());
    assert_eq!(N, c.len());

    if a[0] == 0.0 && c[N - 1] == 0.0 {
        // Tridiagonal matrix is not periodic. Solve using the divisionless Thomas algorithm.
        thomas_divisionless(a, b, c, d)
    } else {
        // Tridiagonal matrix is periodic. Solve using the divisionless Thomas algorithm and the
        // Sherman-Morrison formula.
        // https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

        let mut u = vec![0.0; N];
        let mut v = vec![0.0; N];
        let mut a_dash = a.to_vec();
        let mut b_dash = b.to_vec();
        let mut c_dash = c.to_vec();

        u[0] = -b[0];
        u[N - 1] = c[N - 1];
        v[0] = 1.0;
        v[N - 1] = -a[0] / b[0];

        a_dash[0] = 0.0;
        b_dash[0] -= u[0];
        b_dash[N - 1] -= u[N - 1] * v[N - 1];
        c_dash[N - 1] = 0.0;

        let y = thomas_divisionless(&a_dash, &b_dash, &c_dash, d);
        let q = thomas_divisionless(&a_dash, &b_dash, &c_dash, &u);

        let v_transpose_y: float = v.iter().zip(&y).map(|(&v, &y)| v * y).sum();
        let v_transpose_q: float = v.iter().zip(&q).map(|(&v, &q)| v * q).sum();
        let q_scale = v_transpose_y / (1.0 + v_transpose_q);

        let mut x = vec![0.0; N];
        for i in 0..N {
            x[i] = y[i] - q_scale * q[i];
        }

        x
    }
}

fn thomas_divisionless(a: &[float], b: &[float], c: &[float], f: &[float]) -> Vec<float> {
    // MATLAB code from Program 13, P. 93, Numerical Mathematics - Quarteroni, Sacco and Saler.
    // In this code `a` is the main diagonal, `b` the sub-diagonal, and 'f' the right-hand side.
    //
    // gamma(1) = 1 / a(1);
    // for i = 2:n
    //     gamma(i) = 1 / (a(i) - b(i) * gamma(i-1) * c(i-1));
    // end
    //
    // y(1) = gamma(1) * f(1);
    // for i = 2:n
    //     y(i) = gamma(i) * (f(i) - b(i) * y(i-1));
    // end
    //
    // x(n) = y(n);
    // for i = n-1:-1:1
    //     x(i) = y(i) - gamma(i) * c(i) * x(i+1);
    // end

    let N = a.len();

    let mut gamma = vec![0.0; N];
    gamma[0] = 1.0 / b[0];
    for i in 1..N {
        gamma[i] = 1.0 / (b[i] - a[i] * gamma[i - 1] * c[i - 1]);
    }

    let mut y = vec![0.0; N];
    y[0] = gamma[0] * f[0];
    for i in 1..N {
        y[i] = gamma[i] * (f[i] - a[i] * y[i - 1]);
    }

    let mut x = vec![0.0; N];
    x[N - 1] = y[N - 1];
    for i in (0..N - 1).rev() {
        x[i] = y[i] - gamma[i] * c[i] * x[i + 1];
    }

    x
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

    #[test]
    fn thomas_sherman_morrison_nonperiodic() {
        let mut a = vec![2.0; 5];
        a[0] = 0.0;
        let b = vec![5.0; 5];
        let mut c = vec![3.0; 5];
        c[4] = 0.0;
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let solution = thomas_sherman_morrison(&a, &b, &c, &d);

        let expected = &[
            0.299248120300752,
            -0.165413533834587,
            0.742857142857143,
            -0.127819548872180,
            1.051127819548872,
        ];

        for (&s, &e) in solution.iter().zip(expected) {
            assert!((s - e).abs() < 1e-15, "{} {}", s, e);
        }
    }

    #[test]
    fn thomas_sherman_morrison_periodic() {
        let a = vec![2.0; 5];
        let b = vec![5.0; 5];
        let c = vec![3.0; 5];
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let solution = thomas_sherman_morrison(&a, &b, &c, &d);

        let expected = &[
            -0.427272727272727,
            0.118181818181818,
            0.754545454545455,
            -0.336363636363636,
            1.390909090909091,
        ];

        for (&s, &e) in solution.iter().zip(expected) {
            assert!((s - e).abs() < 1e-15, "{} {}", s, e);
        }
    }
}
