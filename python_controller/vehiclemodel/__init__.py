from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import ode
from scipy.linalg import expm
import typing

from controller import ControllerInput, ControllerOutput

class VehicleState(typing.NamedTuple):
    position: typing.Tuple[float, float] = (0, 0)
    velocity: typing.Tuple[float, float] = (0, 0)
    heading: float = 0
    angular_velocity: float = 0
    steering_angle: float = 0
    steering_angular_velocity: float = 0

    def controller_input(self) -> ControllerInput:
        return ControllerInput(position=self.position, velocity=self.velocity, heading=self.heading)

class VehicleModel(object):
    __metaclass__ = ABCMeta

    def step(self, dt: float, state: VehicleState, control: ControllerOutput) -> VehicleState:
        state = self.state_from_vehicle_state(state)

        r = ode(self.state_equation)
        r.set_integrator("dopri5")
        r.set_initial_value(state, 0)
        r.set_f_params(control)

        r.integrate(r.t + dt)

        if not r.successful():
            raise "unable to integrate (for some reason)"

        return self.state_to_vehicle_state(r.y, control)

    @abstractmethod
    def state_equation(self, t: float, state, control: ControllerInput):
        raise "Must be implemented"

    @abstractmethod
    def state_from_vehicle_state(self, state: VehicleState):
        raise "Must be implemented"

    @abstractmethod
    def state_to_vehicle_state(self, state, control: ControllerOutput) -> VehicleState:
        raise "Must be implemented"

from .directvelocity import DirectVelocityModel
from .pacejkabicycle import PacejkaBicycleModel