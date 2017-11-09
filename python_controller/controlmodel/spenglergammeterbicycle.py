from math import cos, sin, hypot, atan2
import numpy as np
import typing

from controller import ControllerInput, ControllerOutput
from controlmodel import ControlModel

class Params(typing.NamedTuple):
    # Steering slip - negative for oversteer, positive for understeer
    C1: float = 0.1
    # Steering angle coupling - radians turned / meter travelled
    C2: float = 5
    # Duty cycle to acceleration a = Cm1 * throttle
    Cm1: float = 1
    # Cm2 = Cm1 / v_motor_max (i.e. max speed with no air resistance)
    Cm2: float = 0.5
    # Reduced air resistance coefficient (0.5 * rho * A * C_d)
    Cr2: float = 0

class SpenglerGammeterBicycle(ControlModel):
    def __init__(self, params = Params()):
        self.params = params

    def num_states(self):
        return 4

    def num_inputs(self):
        return 2

    def state_equation(self, t: float, y, control):
        # From ETH 2011 MPCC eqn 2.1

        x, y, phi, v = y
        throttle = control[0]
        delta = control[1]

        C1 = self.params.C1
        C2 = self.params.C2
        Cm1 = self.params.Cm1
        Cm2 = self.params.Cm2
        Cr2 = self.params.Cr2

        x_dot = v * cos(phi + C1 * delta)
        y_dot = v * sin(phi + C1 * delta)
        phi_dot = C2 * delta * v

        motor_force = Cm1 * throttle - Cm2 * throttle * v
        air_resistance = -Cr2 * v * v
        cornering_loss = -v * v * delta * delta * C1 * C2

        v_dot = motor_force + air_resistance + cornering_loss

        return [x_dot, y_dot, phi_dot, v_dot]

    def state_from_controller_input(self, state: ControllerInput) -> np.array:
        x,y = state.position
        phi = state.heading
        v = hypot(state.velocity[0], state.velocity[1])
        return np.array([x, y, phi, v])

    def linearise(self, x_0, u_0) -> (np.array, np.array):
        throttle, delta = u_0
        _, _, phi, v = x_0

        C1 = self.params.C1
        C2 = self.params.C2
        Cm1 = self.params.Cm1
        Cm2 = self.params.Cm2
        Cr2 = self.params.Cr2

        sin_k = sin(phi + C1 * delta)
        cos_k = cos(phi + C1 * delta)

        A = np.array([
            [0, 0, -v * sin_k, cos_k],
            [0, 0, v * cos_k, sin_k],
            [0, 0, 0, C2 * delta],
            [0, 0, 0, -Cm2 * throttle - 2 * (Cr2 + C2 * C1 * delta * delta) * v]
        ])

        B = np.array([
            [0, -C1 * v * sin_k],
            [0, C1 * v * cos_k],
            [0, C2 * v],
            [Cm1 - Cm2 * v, -2 * C2 * C1 * v * v * delta]
        ])

        return A, B
