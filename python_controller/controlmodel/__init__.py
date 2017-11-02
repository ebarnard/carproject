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

    @abstractmethod
    def linearise(self, x_0, u_0):
        raise "Must be implemented"

    def discretise(self, dt, x_0, u_0):
        A, B = self.linearise(x_0, u_0)

        A_range = (slice(0, A.shape[0]), slice(0, A.shape[1]))
        B_range = (slice(0, B.shape[0]), slice(A.shape[0], A.shape[0] + B.shape[0]))

        n = A.shape[1] + B.shape[1]
        C = np.zeros((n, n))
        C[A_range] = A
        C[B_range] = B
    
        D = sp.linalg.expm2(dt * C)

        return D[A_range], D[B_range]

from .directvelocity import DirectVelocity