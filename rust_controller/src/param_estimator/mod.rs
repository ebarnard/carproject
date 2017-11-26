use prelude::*;

use controller::{Control, State};

mod fixed_horizon;
pub use self::fixed_horizon::FixedHorizon;

pub trait ParamEstimator {
    fn update(&mut self, dt: float, x0: &State, u: &Control, x: &State) -> &[float];
    fn params(&self) -> &[float];
}
