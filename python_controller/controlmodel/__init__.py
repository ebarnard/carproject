from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp
import typing

from controller import ControllerInput, ControllerOutput
from utils import norm_angle

class ControlModel(object):
    __metaclass__ = ABCMeta

    def step(self, dt: float, state, control):
        fun = lambda t, state: self.state_equation(t, state, control)
        ret = solve_ivp(fun, (0, dt), state, method='RK23', t_eval=[dt])

        if not ret.success:
            raise "unable to integrate (for some reason)"

        return ret.y.ravel()
        
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
        A_d = I + A * dt + A * A * dt * dt / 2
        B_d = np.dot(I * dt + A * dt * dt / 2, B)

        return A_d, B_d

from .directvelocity import DirectVelocity
from .spenglergammeterbicycle import SpenglerGammeterBicycle