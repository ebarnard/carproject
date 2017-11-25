from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
import typing

from controller import ControllerInput, ControllerOutput

class ControlModel(object):
    __metaclass__ = ABCMeta

    def step(self, dt: float, state, control):
        fun = lambda state: self.state_equation(dt, state, control)
        return rk4(state, fun, dt)
        
    @abstractmethod
    def state_equation(self, t: float, state, control):
        raise "Must be implemented"

    @abstractmethod
    def state_from_controller_input(self, state: ControllerInput):
        raise "Must be implemented"

    def control_to_controller_output(self, control):
        return ControllerOutput(throttle_position=control[0], steering_angle=control[1])

    def controller_output_to_control(self, control):
        return [control.throttle_position, control.steering_angle]

    @abstractmethod
    def linearise(self, x_0, u_0):
        raise "Must be implemented"

    def discretise(self, dt, x_0, u_0):
        A, B = self.linearise(x_0, u_0)

        # Second order taylor expansion for exp(x)
        I = np.identity(A.shape[0])
        A_d = I + (A * dt) + (A.dot(A) * dt * dt / 2)
        B_d = np.dot(I * dt + A * dt * dt / 2, B)

        return A_d, B_d

def rk4(x, fx, hs):
    x = np.array(x)
    k1 = np.array(fx(x)) * hs
    xk = x + k1 * 0.5
    k2 = np.array(fx(xk)) * hs
    xk = x + k2 * 0.5
    k3 = np.array(fx(xk)) * hs
    xk = x + k3
    k4 = np.array(fx(x)) * hs
    x = x + (k1 + 2 * (k2 + k3) + k4) / 6

    return x

from .directvelocity import DirectVelocity
from .spenglergammeterbicycle import SpenglerGammeterBicycle