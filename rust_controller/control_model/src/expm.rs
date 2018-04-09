#![allow(non_snake_case)]

use nalgebra::allocator::Allocator;
use nalgebra::storage::Storage;
use nalgebra::{DefaultAllocator, Dim, Matrix, MatrixMN};

use prelude::{from_dynamic, to_dynamic};

// Algorithm from: Higham, Nicholas J. The Scaling and Squaring Method for the Matrix Exponential
// Revisited.
// Based on MatrixBase::exp() from Eigen unsupported.
pub fn expm<D: Dim /* + DimMin<D, Output = D>*/, S: Storage<f64, D, D>>(
    A: &Matrix<f64, D, D, S>,
) -> MatrixMN<f64, D, D>
where
    DefaultAllocator: Allocator<f64, D, D>, /* + Allocator<(usize, usize), D>*/
{
    let (U, V, num_squarings) = compute_uv(A);

    // Pade approximant is (U+V) / (-U+V)
    // Use heap allocated matrices to avoid ridiculous generic bounds.
    // TODO: Use stack allocation once nalgebra supports integer generics.
    let (nrows, ncols) = U.data.shape();
    let mut numer = to_dynamic(&(&U + &V));
    let denom = to_dynamic(&(-U + V));
    if !denom.lu().solve_mut(&mut numer) {
        panic!("system not invertible");
    }
    let mut result = from_dynamic(nrows, ncols, &numer);

    for _ in 0..num_squarings {
        // undo scaling by repeated squaring
        result = &result * &result;
    }
    result
}

fn compute_uv<D: Dim, S: Storage<f64, D, D>>(
    A: &Matrix<f64, D, D, S>,
) -> (MatrixMN<f64, D, D>, MatrixMN<f64, D, D>, u32)
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    let mut l1_norm = 0.0;
    for i in 0..A.ncols() {
        let abs_col_sum = A.column(i).iter().map(|v| v.abs()).sum();
        if abs_col_sum > l1_norm {
            l1_norm = abs_col_sum;
        }
    }

    if l1_norm < 1.495585217958292e-2 {
        let (U, V) = pade3(&A);
        (U, V, 0)
    } else if l1_norm < 2.539398330063230e-1 {
        let (U, V) = pade5(&A);
        (U, V, 0)
    } else if l1_norm < 9.504178996162932e-1 {
        let (U, V) = pade7(&A);
        (U, V, 0)
    } else if l1_norm < 2.097847961257068 {
        let (U, V) = pade9(&A);
        (U, V, 0)
    } else {
        const MAX_NORM: f64 = 5.371920351148152;
        let mut num_squarings = (l1_norm / MAX_NORM).log2().ceil() as i32;
        if num_squarings < 0 {
            num_squarings = 0;
        }
        let A = A.map(|v| v * 2f64.powi(-num_squarings));
        let (U, V) = pade13(A);
        (U, V, num_squarings as u32)
    }
}

fn pade3<D: Dim, S: Storage<f64, D, D>>(
    A: &Matrix<f64, D, D, S>,
) -> (MatrixMN<f64, D, D>, MatrixMN<f64, D, D>)
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    const B: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
    let (nrows, ncols) = A.data.shape();
    let I = MatrixMN::identity_generic(nrows, ncols);

    let A2 = A * A;

    let tmp = B[3] * &A2 + B[1] * &I;
    let U = A * tmp;
    let V = B[2] * A2 + B[0] * I;
    (U, V)
}

fn pade5<D: Dim, S: Storage<f64, D, D>>(
    A: &Matrix<f64, D, D, S>,
) -> (MatrixMN<f64, D, D>, MatrixMN<f64, D, D>)
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    const B: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
    let (nrows, ncols) = A.data.shape();
    let I = MatrixMN::identity_generic(nrows, ncols);

    let A2 = A * A;
    let A4 = &A2 * &A2;

    let tmp = B[5] * &A4 + B[3] * &A2 + B[1] * &I;
    let U = A * tmp;
    let V = B[4] * A4 + B[2] * A2 + B[0] * I;
    (U, V)
}

fn pade7<D: Dim, S: Storage<f64, D, D>>(
    A: &Matrix<f64, D, D, S>,
) -> (MatrixMN<f64, D, D>, MatrixMN<f64, D, D>)
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    const B: [f64; 8] = [
        17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0
    ];
    let (nrows, ncols) = A.data.shape();
    let I = MatrixMN::identity_generic(nrows, ncols);

    let A2 = A * A;
    let A4 = &A2 * &A2;
    let A6 = &A4 * &A2;

    let tmp = B[7] * &A6 + B[5] * &A4 + B[3] * &A2 + B[1] * &I;
    let U = A * tmp;
    let V = B[6] * A6 + B[4] * A4 + B[2] * A2 + B[0] * I;
    (U, V)
}

fn pade9<D: Dim, S: Storage<f64, D, D>>(
    A: &Matrix<f64, D, D, S>,
) -> (MatrixMN<f64, D, D>, MatrixMN<f64, D, D>)
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    const B: [f64; 10] = [
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    ];
    let (nrows, ncols) = A.data.shape();
    let I = MatrixMN::identity_generic(nrows, ncols);

    let A2 = A * A;
    let A4 = &A2 * &A2;
    let A6 = &A4 * &A2;
    let A8 = &A6 * &A2;

    let tmp = B[9] * &A8 + B[7] * &A6 + B[5] * &A4 + B[3] * &A2 + B[1] * &I;
    let U = A * tmp;
    let V = B[8] * A8 + B[6] * A6 + B[4] * A4 + B[2] * A2 + B[0] * I;
    (U, V)
}

fn pade13<D: Dim, S: Storage<f64, D, D>>(
    A: Matrix<f64, D, D, S>,
) -> (MatrixMN<f64, D, D>, MatrixMN<f64, D, D>)
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    const B: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];
    let (nrows, ncols) = A.data.shape();
    let I = MatrixMN::identity_generic(nrows, ncols);

    let A2 = &A * &A;
    let A4 = &A2 * &A2;
    let A6 = &A4 * &A2;

    let V = B[13] * &A6 + B[11] * &A4 + B[9] * &A2; // used for temporary storage
    let tmp = &A6 * V + B[7] * &A6 + B[5] * &A4 + B[3] * &A2 + B[1] * &I;
    let U = A * tmp;

    let tmp = B[12] * &A6 + B[10] * &A4 + B[8] * &A2;
    let V = &A6 * tmp + B[6] * A6 + B[4] * A4 + B[2] * A2 + B[0] * I;
    (U, V)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix2, Matrix3, Matrix4};

    use super::expm;

    #[test]
    fn expm_pade3() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix2::new(
            0.000043000000000, 0.009912000000000,
            0.000012000000000, 0.000000100000000,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = Matrix2::new(
            1.000043060398221, 0.009912213803161,
            0.000012000258842, 1.000000159472862,
        );

        let exp_A = expm(&A);
        if !exp_A.relative_eq(&expected, 1e-15, 1e-15) {
            panic!("expected: {}. actual: {}.", expected, exp_A);
        }
    }

    #[test]
    fn expm_pade5() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix3::new(
            0.08, 0.01, 0.06,
            0.03, 0.05, 0.07,
            0.04, 0.09, 0.02,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = Matrix3::new(
            1.084756909028977, 0.013528665649584, 0.063548668049722,
            0.033536666609639, 1.054744907588895, 0.073552668529749,
            0.043540667089667, 0.093560669489804, 1.024732906148813,
        );

        let exp_A = expm(&A);
        if !exp_A.relative_eq(&expected, 1e-15, 1e-15) {
            panic!("expected: {}. actual: {}.", expected, exp_A);
        }
    }

    #[test]
    fn expm_pade7() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix2::new(
            0.547215529963803, 0.149294005559057,
            0.138624442828679, 0.257508254123736,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = Matrix2::new(
            1.744737713624294, 0.224801146083571,
            0.208735330707944, 1.308507689560578,
        );

        let exp_A = expm(&A);
        if !exp_A.relative_eq(&expected, 1e-15, 1e-15) {
            panic!("expected: {}. actual: {}.", expected, exp_A);
        }
    }

    #[test]
    fn expm_pade9() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix3::new(
            0.190779228546504, 0.093436302277189, 0.323156505055632,
            0.382758394074501, 0.244882197894116, 0.354682415429036,
            0.397599950568532, 0.222793100355450, 0.377343340991180,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = Matrix3::new(
            1.326135792780767, 0.169570426358187, 0.470509231126333,
            0.591507815457227, 1.363243020404054, 0.587777264977557,
            0.608716349482150, 0.342042663730710, 1.613934583308134,
        );

        let exp_A = expm(&A);
        if !exp_A.relative_eq(&expected, 1e-15, 1e-15) {
            panic!("expected: {}. actual: {}.", expected, exp_A);
        }
    }

    #[test]
    fn expm_pade13_no_scaling() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix4::new(
            0.814723686393179, 0.632359246225410, 0.957506835434298, 0.957166948242946,
            0.905791937075619, 0.097540404999410, 0.964888535199277, 0.485375648722841,
            0.126986816293506, 0.278498218867048, 0.157613081677548, 0.800280468888800,
            0.913375856139019, 0.546881519204984, 0.970592781760616, 0.141886338627215,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = Matrix4::new(
            4.720394021183613, 2.456123391535519, 4.183565616967832, 3.673653234120964,
            2.928866714497507, 2.481177196987064, 3.228759552473233, 2.576698942351026,
            1.410035511630756, 1.045823715355211, 2.569074262471745, 1.803629754049430,
            3.039392477683259, 1.909054748938514, 3.348014370985685, 3.374766717641546,
        );

        let exp_A = expm(&A);
        if !exp_A.relative_eq(&expected, 1e-15, 1e-15) {
            panic!("expected: {}. actual: {}.", expected, exp_A);
        }
    }

    #[test]
    fn expm_pade13_scaling() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let A = Matrix4::new(
            14.198665467081877, 16.491708248734184, 22.611872828354858,  1.618503559998214,
            10.549785211889903, 27.515809914894302, 11.413375409260700, 15.923926590269181,
            24.924858836888724,  8.575170564611206, 17.034649221756634, 23.375016903060335,
            17.557922734581727, 22.716006873321639,  2.275628686891908, 28.020320526875491,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = 1e28 * Matrix4::new(
            0.810104285109246, 1.002536192025060, 0.649717791161234, 0.858227260926821,
            0.966444748779387, 1.196013718299542, 0.775105574675043, 1.023854884893233,
            1.072383921402965, 1.327117647337523, 0.860070642135657, 1.136087208085210,
            1.020955411589432, 1.263472826124129, 0.818824078683328, 1.081603668222360,
        );

        let exp_A = expm(&A);
        if !exp_A.relative_eq(&expected, 1e-14, 1e-14) {
            panic!("expected: {}. actual: {}.", expected, exp_A);
        }
    }
}
