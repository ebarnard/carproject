from math import pi, sin, cos, atan, atan2
import numpy as np
from scipy.integrate import ode
import typing

from controller import ControllerOutput
from utils import norm_angle
from vehiclemodel import VehicleModel, VehicleState
import controlmodel

class SpenglerGammeterBicycle(VehicleModel):
    def __init__(self, model = controlmodel.SpenglerGammeterBicycle()):
        self.model = model

    def state_equation(self, t: float, state, control: ControllerOutput):
        return self.model.state_equation(t, state, self.model.controller_output_to_control(control))

    def state_from_vehicle_state(self, state: VehicleState):
        return self.model.state_from_controller_input(state.controller_input())

    def state_to_vehicle_state(self, state, control: ControllerOutput) -> VehicleState:
        heading = state[2]
        v = state[3]

        adj_heading = heading + self.model.params.C2 * control.steering_angle
        v_x = v * cos(adj_heading)
        v_y = v * sin(adj_heading)

        return VehicleState(
            position = (state[0], state[1]),
            heading = heading,
            velocity = (v_x, v_y)
        )
