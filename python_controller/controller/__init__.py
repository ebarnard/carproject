from abc import ABCMeta, abstractmethod
import numpy as np
import typing

class ControllerInput(typing.NamedTuple):
    position: typing.Tuple[float, float]
    velocity: typing.Tuple[float, float]
    heading: float

class ControllerOutput(typing.NamedTuple):
    throttle_position: float
    steering_angle: float

class Controller(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(dt: float, state: ControllerInput) -> ControllerOutput:
        pass

# Local reexports
from .condensedqpbuilder import CondensedQPBuilder
from .lookaheadproportional import LookaheadProportionalController
from .mpcpositionvelocity import MPCPositionVelocityController

