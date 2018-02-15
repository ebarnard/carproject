use config;

use prelude::*;
use control_model::{self, Control, ControlModel, State as ControllerState};

#[derive(Clone, Debug, Default)]
pub struct State {
    pub position: (float, float),
    pub velocity: (float, float),
    pub heading: float,
}

impl State {
    pub fn to_controller_state(&self) -> ControllerState {
        ControllerState {
            position: self.position,
            velocity: self.velocity,
            heading: self.heading,
        }
    }
}

pub trait SimulationModel {
    fn new(params: &[float]) -> Self
    where
        Self: Sized;

    fn step(&mut self, dt: float, state: &State, control: &Control) -> State;
}

pub fn model_from_config(config: &config::Simulator) -> Box<SimulationModel> {
    MODELS
        .iter()
        .find(|m| m.0 == config.model)
        .expect("simulation model not found")
        .1(&config.params)
}

static MODELS: &'static [(&'static str, fn(&[float]) -> Box<SimulationModel>)] = &[
    ("direct_velocity", new::<DirectVelocity>),
    ("spengler_gammeter_bicycle", new::<SpenglerGammeterBicycle>),
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

            fn step(&mut self, dt: float, state: &State, control: &Control) -> State {
                let x = self.model.x_from_state(&state.to_controller_state());
                let u = self.model.u_from_control(control);
                let x_dt = self.model.step(dt, &x, &u, &self.params);
                let controller_state = self.model.x_to_state(&x_dt);

                State {
                    position: controller_state.position,
                    velocity: controller_state.velocity,
                    heading: controller_state.heading,
                }
            }
        }
    );
);

simulation_control_model!(DirectVelocity, control_model::DirectVelocity);
simulation_control_model!(
    SpenglerGammeterBicycle,
    control_model::SpenglerGammeterBicycle
);
