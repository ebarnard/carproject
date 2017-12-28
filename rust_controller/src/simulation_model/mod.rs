use prelude::*;
use controller::{Control, State as ControllerState};
use control_model::{self, ControlModel};

#[derive(Debug, Default)]
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
    fn step(&mut self, dt: float, state: &State, control: &Control) -> State;
}

macro_rules! simulation_control_model(
    ($name:ident) => (
        pub struct $name {
            model: control_model::$name,
            params: Vector<<control_model::$name as ControlModel>::NP>,
        }

        impl $name {
            pub fn new(params: Vector<<control_model::$name as ControlModel>::NP>) -> Self {
                $name {
                    params,
                    model: control_model::$name,
                }
            }
        }

        impl SimulationModel for $name {
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

simulation_control_model!(DirectVelocity);
simulation_control_model!(SpenglerGammeterBicycle);
