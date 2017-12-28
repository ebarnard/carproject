use prelude::*;
use control_model::ControlModel;
use simulation_model::State;

mod ekf;
pub use self::ekf::EKF;
pub use self::ekf::JointEKF;

mod param_least_squares;

pub trait Estimator<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn step(
        &mut self,
        model: &M,
        dt: float,
        u: &Vector<M::NI>,
        measure: Option<Measurement>,
        p: &Vector<M::NP>,
    ) -> (Vector<M::NS>, Vector<M::NP>);

    fn param_covariance(&self) -> Matrix<M::NP, M::NP>;
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
