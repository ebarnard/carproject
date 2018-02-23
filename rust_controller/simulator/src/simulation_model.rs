use prelude::*;
use control_model::{self, Control, ControlModel, State};

use config::SimulatorConfig;

#[derive(Clone)]
pub struct SimState(Vec<float>);

pub trait SimulationModel {
    fn new(params: &[float]) -> Self
    where
        Self: Sized;

    fn name() -> &'static str
    where
        Self: Sized;

    fn init_state(&self, x: float, y: float, heading: float) -> SimState;

    fn inspect_state(&self, state: &SimState) -> State;

    fn step(&mut self, dt: float, state: SimState, control: &Control) -> SimState;
}

pub fn model_from_config(config: &SimulatorConfig) -> Box<SimulationModel> {
    MODELS
        .iter()
        .find(|m| m.0() == config.model)
        .expect("simulation model not found")
        .1(&config.params)
}

static MODELS: &'static [(fn() -> &'static str, fn(&[float]) -> Box<SimulationModel>)] = &[
    (DirectVelocity::name, new::<DirectVelocity>),
    (
        SpenglerGammeterBicycle::name,
        new::<SpenglerGammeterBicycle>,
    ),
    (NoSlipBicycle::name, new::<NoSlipBicycle>),
];

fn new<T: 'static + SimulationModel>(params: &[float]) -> Box<SimulationModel> {
    Box::new(T::new(params))
}

macro_rules! simulation_control_model(
    ($name:ident, $model:ty) => (
        pub struct $name {
            model: $model,
            params: Vector<<$model as ControlModel>::NP>,
        }

        impl SimulationModel for $name {
            fn new(params: &[float]) -> Self
            where
                Self: Sized
            {
                $name {
                    params: Vector::<<$model as ControlModel>::NP>::from_column_slice(params),
                    model: <$model as ControlModel>::new(),
                }
            }

            fn name() -> &'static str {
                <$model as ControlModel>::name()
            }

            fn init_state(&self, x: float, y: float, heading: float) -> SimState {
                let mut state = vec![0.0; <$model as ControlModel>::NS::dim()];
                state[0] = x;
                state[1] = y;
                state[2] = heading;
                SimState(state)
            }

            fn inspect_state(&self, state: &SimState) -> State {
                let x = Vector::<<$model as ControlModel>::NS>::from_column_slice(&state.0);
                self.model.x_to_state(&x)
            }

            fn step(&mut self, dt: float, mut state: SimState, control: &Control) -> SimState {
                let x = Vector::<<$model as ControlModel>::NS>::from_column_slice(&state.0);
                let u = self.model.u_from_control(control);
                let x_dt = self.model.step(dt, &x, &u, &self.params);
                state.0.copy_from_slice(x_dt.as_slice());
                state
            }
        }
    );
);

simulation_control_model!(DirectVelocity, control_model::DirectVelocity);
simulation_control_model!(
    SpenglerGammeterBicycle,
    control_model::SpenglerGammeterBicycle
);
simulation_control_model!(NoSlipBicycle, control_model::NoSlipBicycle);
