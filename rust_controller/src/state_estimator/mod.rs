use prelude::*;
use control_model::ControlModel;
use simulation_model::State;

mod ekf;
pub use self::ekf::EKF;

pub trait StateEstimator<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> Vector<M::NS>;
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
