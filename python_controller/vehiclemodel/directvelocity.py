from math import pi, sin, cos, atan, atan2
import numpy as np
from scipy.integrate import ode
import typing

from controller import ControllerOutput
from utils import norm_angle
from vehiclemodel import VehicleModel, VehicleState

class DirectVelocity(VehicleModel):
    def state_from_vehicle_state(self, state: VehicleState):
        return [state.position[0], state.position[1], state.heading]

    def state_to_vehicle_state(self, state, control: ControllerOutput) -> VehicleState:
        X_dt, Y_dt, phi_dt = state

        # Normalize heading to be between -pi and pi
        #phi_dt = norm_angle(phi_dt)
        v_x = control.throttle_position * cos(phi_dt)
        v_y = control.throttle_position * sin(phi_dt)

        return VehicleState(position=(X_dt, Y_dt), velocity=(v_x, v_y), heading=phi_dt)

    def state_equation(self, t: float, y, control: ControllerOutput):
        # From ETH 2011 MPCC eqn 2.1
        
        x, y, phi = y

        v = control.throttle_position
        delta = control.steering_angle

        x_dot = v * cos(phi)
        y_dot = v * sin(phi)
        phi_dot = delta

        return [x_dot, y_dot, phi_dot]

