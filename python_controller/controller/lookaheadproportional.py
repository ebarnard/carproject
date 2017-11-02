from math import atan2, hypot
from scipy.spatial import KDTree

from controller import Controller, ControllerInput, ControllerOutput
from track import Track
from utils import norm_angle

class LookaheadProportionalController(Controller):
    def __init__(self, trk: Track):
        self.track = trk
        self.current_index = 0
        self.lookahead = 5

    def step(self, dt: float, state: ControllerInput) -> ControllerOutput:
        current_position = state.position

        # Find the nearest centerline point
        def dist(i):
            return (self.track.x()[i] - current_position[0]) ** 2 + (self.track.y()[i] - current_position[1]) ** 2

        i = self.current_index
        best_distance = dist(i)
        while True:
            next_i = (i + 1) % len(self.track.x())
            distance = dist(next_i)
            if distance > best_distance:
                break
            best_distance = distance
            i = next_i
        self.current_index = i
        
        # # and the point lookahead steps
        i = (i + self.lookahead) % len(self.track.x())
        target_position = self.track.xy()[:,i]

        # Point the car to look at target_position
        # Set throttle as some function of diff_angle
        target_angle = atan2(target_position[1] - current_position[1], target_position[0] - current_position[0])
        velocity_angle = atan2(state.velocity[1], state.velocity[0])

        error_angle = norm_angle(target_angle - velocity_angle)

        steering_angle = max(min(error_angle, 0.6), -0.6)

        v_ref = 0.1
        v = hypot(state.velocity[0], state.velocity[1])
        throttle_position = max(min((v_ref - v) * 0.5, 1), 0)

        return ControllerOutput(throttle_position = throttle_position, steering_angle = steering_angle)