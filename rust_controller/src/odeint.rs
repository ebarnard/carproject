use prelude::*;

pub fn rk4<N: DimName, F>(dt: float, num_steps: u32, y_0: &Vector<N>, mut f: F) -> Vector<N>
where
    F: FnMut(&Vector<N>) -> Vector<N>,
    DefaultAllocator: Allocator<float, N>,
{
    let h = dt / float::from(num_steps);
    let mut y = y_0.clone();
    for _ in 0..num_steps {
        let k1 = f(&y) * h;
        let k2 = f(&(&y + 0.5 * &k1)) * h;
        let k3 = f(&(&y + 0.5 * &k2)) * h;
        let k4 = f(&(&y + &k3)) * h;
        y += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
    }
    y
}
