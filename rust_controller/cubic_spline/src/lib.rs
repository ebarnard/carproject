#![allow(non_snake_case)]

extern crate prelude;

use prelude::*;

#[derive(Clone)]
pub struct CubicSpline {
    coefs: Vec<float>,
}

impl CubicSpline {
    pub fn periodic(y: &[float]) -> CubicSpline {
        // Maths from: http://mathworld.wolfram.com/CubicSpline.html

        let n = y.len();
        assert!(n >= 2);

        // Allocate vector that will be used to store spline coefficients. This is the only
        // allocation made by this function.
        let mut coefs = vec![0.0; n * 4];

        {
            // Store the spline matrix equation right-hand side in the second `n` elements of
            // `coefs`, use the next `n` elements as scratch space and store the output in the
            // final `n` elements.
            let (left, right) = coefs.split_at_mut(n * 2);
            let (_, rhs) = left.split_at_mut(n);
            let (zeros, D) = right.split_at_mut(n);

            rhs[0] = 3.0 * (y[1] - y[n - 1]);
            for i in 1..n - 1 {
                rhs[i] = 3.0 * (y[i + 1] - y[i - 1]);
            }
            rhs[n - 1] = 3.0 * (y[0] - y[n - 2]);

            thomas_sherman_morrison(1.0, 4.0, 1.0, rhs, D, zeros);
        }

        let make_coef = |y, y_p, D, D_p| {
            let a = y;
            let b = D;
            let c = 3.0 * (y_p - y) - 2.0 * D - D_p;
            let d = 2.0 * (y - y_p) + D + D_p;
            [a, b, c, d]
        };

        // The final `n` elements of `coefs` now contains the `D` vector.
        let D_0 = coefs[n * 3];
        for i in 0..n - 1 {
            let coef = make_coef(y[i], y[i + 1], coefs[n * 3 + i], coefs[n * 3 + i + 1]);
            (&mut coefs[i * 4..(i + 1) * 4]).copy_from_slice(&coef);
        }
        let coef = make_coef(y[n - 1], y[0], coefs[n * 4 - 1], D_0);
        (&mut coefs[(n - 1) * 4..n * 4]).copy_from_slice(&coef);

        CubicSpline { coefs }
    }

    fn get_coefs(&self, t: float) -> (float, float, float, float) {
        let i = (t.floor() as usize) % self.coefs.len();
        let coef = &self.coefs[i * 4..(i + 1) * 4];
        (coef[0], coef[1], coef[2], coef[3])
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

/// Returns the first row of the symmetric and circulant periodic cubic spline second derivative
/// matrix:
///
/// ```text
/// │ x''(t = 0) │   │ a  b  c  d  e  d  c  b │   │ x(t = 0) │
/// │ x''(t = 1) │   │ b  a  b  c  d  e  d  c │   │ x(t = 1) │
/// │ x''(t = 2) │   │ c  b  a  b  c  d  e  d │   │ x(t = 2) │
/// │ x''(t = 3) │ = │ d  c  b  a  b  c  d  e │ * │ x(t = 3) │
/// │ x''(t = 4) │   │ e  d  c  b  a  b  c  d │   │ x(t = 4) │
/// │ x''(t = 5) │   │ d  e  d  c  b  a  b  c │   │ x(t = 5) │
/// │ x''(t = 6) │   │ c  d  e  d  c  b  a  b │   │ x(t = 6) │
/// │ x''(t = 7) │   │ b  c  d  e  d  c  b  a │   │ x(t = 7) │
/// ```
///
/// The magnitide of the nth element goes as approximately `10 ^ (-n / 2)`.
///
pub fn periodic_second_derivative_matrix(vals: &mut [float]) {
    let N = vals.len();
    assert!(N >= 3);

    let mut zeros = vec![0.0; N * 2];
    let (rhs, zeros) = zeros.split_at_mut(N);

    rhs[0] = -12.0;
    rhs[1] = 6.0;
    rhs[N - 1] = 6.0;

    thomas_sherman_morrison(1.0, 4.0, 1.0, rhs, vals, zeros);
}

/// Solves a tridiagonal linear system of the form:
///
/// ```text
/// │ b  c  0  0  a │   | x0 │   │ d0 │
/// │ a  b  c  0  0 │   | x1 │   │ d1 │
/// │ 0  a  b  c  0 │ * | x2 │ = │ d2 │
/// │ 0  0  a  b  c │   | x3 │   │ d3 │
/// │ c  0  0  a  b │   | x4 │   │ d4 │
/// ```
///
/// The `d` and `zeros` vectors are used as scratch space. The `zeros` vector is expected to
/// contain only zeros.
fn thomas_sherman_morrison(
    a: float,
    b: float,
    c: float,
    d: &mut [float],
    x: &mut [float],
    zeros: &mut [float],
) {
    // Tridiagonal matrix is periodic. Solve using the divisionless Thomas algorithm and the
    // Sherman-Morrison formula.
    // https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

    let N = d.len();
    assert!(N >= 3);
    assert_eq!(x.len(), N);
    assert_eq!(zeros.len(), N);

    let u = zeros;
    u[0] = -b;
    u[N - 1] = c;
    let v_0 = 1.0;
    let v_N = -a / b;

    let b_0 = b - u[0];
    let b_N = b - u[N - 1] * v_N;

    let q = x;
    thomas_divisionless(a, b_0, b, b_N, c, u, q);
    let v_transpose_q = v_0 * q[0] + v_N * q[N - 1];

    let y = u;
    thomas_divisionless(a, b_0, b, b_N, c, d, y);
    let v_transpose_y = v_0 * y[0] + v_N * y[N - 1];

    let q_scale = v_transpose_y / (1.0 + v_transpose_q);

    let x = q;
    for i in 0..N {
        x[i] = y[i] - q_scale * x[i];
    }
}

/// Solves a tridiagonal linear system of the form:
///
/// ```text
/// │ b0  c  0  0  0  │   │ x0 │   │ d0 │
/// │  a  b  c  0  0  │   │ x1 │   │ d1 │
/// │  0  a  b  c  0  │ * │ x2 │ = │ d2 │
/// │  0  0  a  b  c  │   │ x3 │   │ d3 │
/// │  0  0  0  a  bN │   │ x4 │   │ d4 │
/// ```
///
/// The `d` vector is used as scratch space.
fn thomas_divisionless(
    a: float,
    b_0: float,
    b: float,
    b_N: float,
    c: float,
    d: &mut [float],
    x: &mut [float],
) {
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

    let N = d.len();
    assert!(N >= 3);
    assert_eq!(x.len(), N);

    let a_c = a * c;

    // Use no longer required entries of `d` to store the `gamma` vector.
    let gamma_0 = 1.0 / b_0;
    x[0] = gamma_0 * d[0];
    d[0] = gamma_0;
    for i in 1..(N - 1) {
        let gamma_i = 1.0 / (b - a_c * d[i - 1]);
        x[i] = gamma_i * (d[i] - a * x[i - 1]);
        d[i] = gamma_i;
    }
    let gamma_N_1 = 1.0 / (b_N - a_c * d[N - 2]);
    x[N - 1] = gamma_N_1 * (d[N - 1] - a * x[N - 2]);

    let gamma = d;
    for i in (0..N - 1).rev() {
        x[i] -= gamma[i] * c * x[i + 1];
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matlab_d_values() {
        let y = &[5., 6., 8., 2., 1.5, 7.4, 9.];

        let spline = CubicSpline::periodic(y);
        let D: Vec<_> = spline.coefs.chunks(4).map(|c| c[1]).collect();

        let D_expected = &[
            -2.509756097560976,
            3.519512195121950,
            -2.568292682926829,
            -5.246341463414635,
            4.053658536585368,
            5.231707317073170,
            -2.480487804878049,
        ];

        assert_approx_eq(&D, D_expected, 1e-15);
    }

    #[test]
    fn matlab_second_derivative_matrix() {
        let mut x = vec![0.0; 2500];

        periodic_second_derivative_matrix(&mut x);

        let x_expected = &[
            -4.392304845413264,
            2.784609690826527,
            -0.746133917892846,
            0.199925980744858,
            -0.053570005086585,
            0.014354039601482,
            -0.003846153319341,
            0.001030573675884,
            -0.000276141384194,
            0.000073991860892,
            -0.000019826059372,
            0.000005312376598,
            -0.000001423447019,
            0.000000381411479,
            -0.000000102198898,
            0.000000027384112,
            -0.000000007337551,
            0.000000001966091,
            -0.000000000526812,
            0.000000000141159,
            -0.000000000037823,
            0.000000000010135,
            -0.000000000002716,
            0.000000000000728,
            -0.000000000000195,
            0.000000000000052,
            -0.000000000000014,
            0.000000000000004,
            -0.000000000000001,
            0.000000000000000,
        ];

        assert_approx_eq(&x, x_expected, 1e-15);
    }

    #[test]
    fn thomas_nonperiodic() {
        let a = 2.0;
        let b_0 = 10.0;
        let b = 5.0;
        let b_N = 15.0;
        let c = 3.0;
        let mut d = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut x = vec![0.0; 5];
        thomas_divisionless(a, b_0, b, b_N, c, &mut d, &mut x);

        let x_expected = &[
            -0.002966101694915,
            0.343220338983051,
            0.096610169491525,
            0.610169491525424,
            0.251977401129943,
        ];

        assert_approx_eq(&x, x_expected, 1e-15);
    }

    #[test]
    fn thomas_sherman_morrison_periodic() {
        let a = 2.0;
        let b = 5.0;
        let c = 3.0;
        let mut d = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut x = vec![0.0; 5];
        let mut zeros = vec![0.0; 5];
        thomas_sherman_morrison(a, b, c, &mut d, &mut x, &mut zeros);

        let x_expected = &[
            -0.427272727272727,
            0.118181818181818,
            0.754545454545455,
            -0.336363636363636,
            1.390909090909091,
        ];

        assert_approx_eq(&x, x_expected, 1e-15);
    }

    fn assert_approx_eq(x: &[float], x_expected: &[float], eps: float) {
        for (&x, &x_exp) in x.iter().zip(x_expected) {
            assert!(
                (x - x_exp).abs() < eps,
                "acutal: {}. expected: {}.",
                x,
                x_exp
            );
        }
    }
}
