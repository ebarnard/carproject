use prelude::*;
use controller::{Control, State as ControllerState};
use simulation_model::State;

mod ekf;
pub use self::ekf::EKF;

pub trait StateEstimator {
    fn step(
        &mut self,
        dt: float,
        control: &Control,
        measure: Option<Measurement>,
        params: &[float],
    ) -> ControllerState;
}

pub struct Measurement {
    pub position: (float, float),
    pub heading: float,
}

impl Measurement {
    pub fn from_state(state: &State) -> Measurement {
        Measurement {
            position: state.position,
            heading: state.heading,
        }
    }
}
