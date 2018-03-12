pub extern crate flame;
pub extern crate nalgebra;
extern crate rand;

pub mod flame_merge;

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

use std::time::Duration;
pub fn duration_to_secs(d: Duration) -> float {
    d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9
}

pub fn secs_to_duration(secs: float) -> Duration {
    Duration::new(secs.floor() as u64, (secs.fract() * 1e9) as u32)
}

pub type Matrix<R, C> = nalgebra::MatrixMN<float, R, C>;
pub type Vector<N> = nalgebra::VectorN<float, N>;

use nalgebra::{Dim, DimNameSum, Scalar, U1, U3};
pub use nalgebra::allocator::Allocator;
pub use nalgebra::{DefaultAllocator, DimName, DimNameAdd, Dynamic as Dy};

pub trait ModelDims<A: DimName, B: DimNameAdd<A>, C: DimNameAdd<A>>
    : Dims2<A, B> + Dims2<A, C> + Dims2<B, C> + Dims2<DimNameSum<B, A>, C> + Dims2<DimNameSum<C, A>, B>
    {
}

impl<A, B, C> ModelDims<A, B, C> for DefaultAllocator
where
    A: DimName,
    B: DimNameAdd<A>,
    C: DimNameAdd<A>,
    DefaultAllocator: Dims2<A, B>
        + Dims2<A, C>
        + Dims2<B, C>
        + Dims2<DimNameSum<B, A>, C>
        + Dims2<DimNameSum<C, A>, B>,
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
    + Allocator<N, U1, B>
    + Allocator<N, A, U3>
    + Allocator<N, B, U3>
    + Allocator<N, U3, A>
    + Allocator<N, U3, B> {
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
        + Allocator<N, U1, B>
        + Allocator<N, A, U3>
        + Allocator<N, B, U3>
        + Allocator<N, U3, A>
        + Allocator<N, U3, B>,
{
}
