pub extern crate flame;
pub extern crate nalgebra;
extern crate rand;

#[allow(non_camel_case_types)]
pub type float = f64;
pub use std::f64::{INFINITY, NEG_INFINITY};
pub use std::f64::consts::PI;

pub fn min<T: Copy + PartialOrd>(a: T, b: T) -> T {
    *nalgebra::partial_min(&a, &b).expect("NaN")
}

pub fn max<T: Copy + PartialOrd>(a: T, b: T) -> T {
    *nalgebra::partial_max(&a, &b).expect("NaN")
}

pub fn phase_unwrap(a: float, mut b: float) -> float {
    if a.is_infinite() || b.is_infinite() {
        return b;
    }
    while b > a + PI {
        b -= 2.0 * PI;
    }
    while b < a - PI {
        b += 2.0 * PI;
    }
    b
}

pub fn randn() -> float {
    let rand::distributions::normal::StandardNormal(n) = rand::random();
    n
}

pub type Matrix<R, C> = nalgebra::MatrixMN<float, R, C>;
pub type Vector<N> = nalgebra::VectorN<float, N>;

use nalgebra::{Dim, Scalar, U1};
pub use nalgebra::allocator::Allocator;
pub use nalgebra::{DefaultAllocator, DimName, Dynamic as Dy};

pub trait Dims3<A: Dim, B: Dim, C: Dim>: Dims2<A, B> + Dims2<A, C> + Dims2<B, C> {}

impl<A, B, C> Dims3<A, B, C> for DefaultAllocator
where
    A: Dim,
    B: Dim,
    C: Dim,
    DefaultAllocator: Dims2<A, B> + Dims2<A, C> + Dims2<B, C>,
{
}

pub trait Dims2<A: Dim, B: Dim>
    : Dims2N<float, A, B> + Dims2N<usize, A, B> + Dims2N<bool, A, B> {
}

impl<A, B> Dims2<A, B> for DefaultAllocator
where
    A: Dim,
    B: Dim,
    DefaultAllocator: Dims2N<float, A, B> + Dims2N<usize, A, B> + Dims2N<bool, A, B>,
{
}

pub trait Dims2N<N: Scalar, A: Dim, B: Dim>
    : Allocator<N, A, A>
    + Allocator<N, B, B>
    + Allocator<N, A, B>
    + Allocator<N, B, A>
    + Allocator<N, A>
    + Allocator<N, B>
    + Allocator<N, U1, A>
    + Allocator<N, U1, B> {
}

impl<N, A, B> Dims2N<N, A, B> for DefaultAllocator
where
    N: Scalar,
    A: Dim,
    B: Dim,
    DefaultAllocator: Allocator<N, A, A>
        + Allocator<N, B, B>
        + Allocator<N, A, B>
        + Allocator<N, B, A>
        + Allocator<N, A>
        + Allocator<N, B>
        + Allocator<N, U1, A>
        + Allocator<N, U1, B>,
{
}
