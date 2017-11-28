use prelude::*;
use control_model::ControlModel;

mod fixed_horizon;
pub use self::fixed_horizon::FixedHorizon;

pub trait ParamEstimator<M: ControlModel>
where
    DefaultAllocator: Dims3<M::NS, M::NI, M::NP>,
{
    fn update(
        &mut self,
        dt: float,
        x0: &Vector<M::NS>,
        u: &Vector<M::NI>,
        x: &Vector<M::NS>,
    ) -> Vector<M::NP>;
}
