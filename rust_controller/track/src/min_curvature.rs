use nalgebra::U1;
use std::iter::once;

use cubic_spline;
use osqp::{Problem, Settings, Status};
use prelude::*;
use sparse;

use Track;

pub fn minc(track: &Track) -> Vec<float> {
    let n = track.cumulative_distance.len();

    // Allocate space for the vectors used in the problem.
    let mut x = Vector::zeros_generic(Dy::new(n), U1);
    let mut y = Vector::zeros_generic(Dy::new(n), U1);
    let mut dx = Vector::zeros_generic(Dy::new(n), U1);
    let mut dy = Vector::zeros_generic(Dy::new(n), U1);
    let mut w = Vector::zeros_generic(Dy::new(n), U1);
    let mut l = Vector::zeros_generic(Dy::new(n), U1);
    let mut u = Vector::zeros_generic(Dy::new(n), U1);

    // Generate the fixed part of the problem matrix mapping x and y coordinates to squared
    // distance and curvature.
    let E_squared_row = generate_problem_row(n, 1.0);

    // Find `n` equally spaced points around the current iteration of the track raceline.
    for (i, &s) in once(&0.0)
        .chain(track.cumulative_distance.iter().take(n - 1))
        .enumerate()
    {
        let point = track.nearest_centreline_point(s);
        x[i] = point.x;
        y[i] = point.y;
        dx[i] = -point.dy_ds;
        dy[i] = point.dx_ds;
        l[i] = -0.5 * point.track_width;
        u[i] = 0.5 * point.track_width;
        w[i] = point.track_width;
    }

    // Generate the H and f matrices for the quadratic program from the previously calculated
    // equally spaced points.

    // The bracket operator [a] creates a diagonal matrix containing the elements of vector a.
    // H = [dx]^T*C^T*C*[dx] + [dy]^T*C^T*C*[dy]
    // H[r, c] = C_squared_row[abs(r - c)] * (dx[r] * dx[c] + dy[r] * dy[c])
    let H = sparse::from_fn(n, n, |r, c| {
        let i = max(r, c) - min(r, c);
        E_squared_row[i] * (dx[r] * dx[c] + dy[r] * dy[c])
    }).build_csc();

    // F = [dx]' * C' * C * x + [dy]' * C' * C * y
    // F[i] = SUM j { C_squared_row[j - i] * (dx[i] * x[j] + dy[i] * y[j]) }
    let f: Vec<float> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let k = if j >= i { j - i } else { n + j - i };
                    E_squared_row[k] * (dx[i] * x[j] + dy[i] * y[j])
                })
                .sum()
        })
        .collect();

    let A = sparse::eye(n).build_csc();

    let settings = Settings::default()
        .eps_abs(1e-11)
        .eps_rel(1e-11)
        .max_iter(20000);

    // TODO: Find sparsity structure of H so we don't continuously reinit
    let mut prob = Problem::new(&H, f.as_slice(), &A, l.as_slice(), u.as_slice(), &settings);

    let alphas = match prob.solve() {
        Status::Solved(s) | Status::SolvedInaccurate(s) | Status::MaxIterationsReached(s) => s.x(),
        _ => panic!("solve failed"),
    };

    for i in 0..100 {
        // Create H and f
        // Attempt to solve problem
        // Check for inf-norm eps tolerance
        // Update x, y, dx, and dy
    }

    alphas.to_vec()
}

fn generate_problem_row(n: usize, nu: float) -> Vec<float> {
    // C is a symmetric and circulant matrix such that x''=Cx and y''=Cy. C_row is the first row of
    // this matrix.
    let mut C_row = vec![0.0; n];
    cubic_spline::periodic_second_derivative_matrix(&mut C_row);
    C_row
        .iter_mut()
        .filter(|&&mut v| v.abs() < 1e-5)
        .for_each(|v| *v = 0.0);

    // C_squared = C^T*C is also a symmetric and circulant matrix. C_squared_row is the first row
    // of this matrix where the ith element is SUM j { C_row[j] * C_row[i + j] }.
    let C_squared_row: Vec<float> = (0..n)
        .map(|i| {
            C_row
                .iter()
                .zip(C_row.iter().cycle().skip(i))
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect();

    // D is a circulant and banded matrix such that the difference between consecutive elements of
    // x is Dx. i.e.
    //     | -1  1  0  0 |
    // D = |  0 -1  1  0 |
    //     |  0  0 -1  1 |
    //     |  1  0  0 -1 |
    // D_squared_row is the first row of D^T*D and is constructed directly.
    let mut D_squared_row = vec![0.0; n];
    D_squared_row[0] = 2.0;
    D_squared_row[1] = -1.0;
    D_squared_row[n - 1] = -1.0;

    // `nu` is the relative weighting for minimising curvature and 'mu' is the weighting for
    // minimising distance.
    let mu = 1.0 - nu;

    let mut E_squared_row = vec![0.0; n];
    for i in 0..n {
        E_squared_row[i] = nu * C_squared_row[i] + mu * D_squared_row[i];
    }
    E_squared_row
}

fn equispace_points() {
    //2.5k points -> many more subdivisions... :-(
    //strategy: recurse until gradient at start + end are almost the same
    // this implies a mostly straight line such that t is prop to s
    // y = a + bx + cx^2 + dx^3
    // y' = b + 2cx + 3dx^2 (varies by 2c + 3d -> if small then below is prolly small...)
    // y'' = 2c + 6dx (change in curvature should also be small...) i.e. 6d small
}

    /*// Write out problem for MATLAB comparison
    use std::io::{BufWriter, Write};
    let mut out = BufWriter::new(::std::fs::File::create("/Users/Edward/qp.csv").unwrap());
    write!(out, "x,y,dx,dy,l,u,f,H\n").unwrap();
    let H_d = sparse::block(&H).to_dense();
    for i in 0..n {
        write!(
            out,
            "{},{},{},{},{},{},{}",
            x[i], y[i], dx[i], dy[i], l[i], u[i], f[i]
        ).unwrap();
        for j in 0..n {
            write!(out, ",{}", H_d[(i, j)]).unwrap();
        }
        write!(out, "\n").unwrap();
    }
    out.flush().unwrap();*/
