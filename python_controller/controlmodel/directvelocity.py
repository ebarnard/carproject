from math import cos, sin, hypot, atan2
import numpy as np

from controller import ControllerInput, ControllerOutput
from controlmodel import ControlModel

class DirectVelocity(ControlModel):
    def state_equation(self, t: float, y, control):
        # From ETH 2011 MPCC eqn 2.1

        x, y, phi = y
        v = control[0]
        delta = control[1]

        x_dot = v * cos(phi)
        y_dot = v * sin(phi)
        phi_dot = delta

        return [x_dot, y_dot, phi_dot]

    def state_from_controller_input(self, state: ControllerInput) -> np.array:
        x,y = state.position
        phi = state.heading
        return np.array([x, y, phi])

    # Takes a point to linearise around (VehicleState)
    # Returns jacobians A and B and x0
    def linearise(self, x_0, u_0) -> (np.array, np.array):
        v_0 = u_0[0]
        phi_0 = x_0[2]

        sin_phi_0 = sin(phi_0)
        cos_phi_0 = cos(phi_0)

        A = np.array([
            [0, 0, -v_0 * sin_phi_0],
            [0, 0, v_0 * cos_phi_0],
            [0, 0, 0]
        ])

        B = np.array([
            [cos_phi_0, 0],
            [sin_phi_0, 0],
            [0, 1]
        ])

        return A, B

    def discretise(self, dt: float, x_0, u_0) -> (np.array, np.array):
        A, B = self.linearise(x_0, u_0)
        I = np.identity(3)

        A_d = I + A * dt
        B_d = np.dot(I * dt + A * dt * dt / 2, B)
        
        return A_d, B_d
