from math import atan2, sin, cos
import numpy as np
import scipy as sp
from timeit import default_timer as timer
import typing

from controller import ControllerInput, ControllerOutput, MPCPositionVelocityController
import controlmodel
import vehiclemodel
from vehiclemodel import VehicleState
import visualisation

class SimHistory(typing.NamedTuple):
    n: int
    positions: np.ndarray
    velocities: np.ndarray
    headings: np.ndarray
    angular_velocities: np.ndarray
    steering_angles: np.ndarray
    control_throttle_positions: np.ndarray
    control_steering_angles: np.ndarray

    def empty(n: int):
        return SimHistory(
            n = n,
            positions = np.empty((2, n)),
            velocities = np.empty((2, n)),
            headings = np.empty((1, n)),
            angular_velocities = np.empty((1, n)),
            steering_angles = np.empty((1, n)),
            control_throttle_positions = np.empty((1, n)),
            control_steering_angles = np.empty((1, n))
        )

    def record(self, i: int, state: VehicleState, control: ControllerOutput):
        self.positions[:,i] = state.position
        self.velocities[:,i] = state.velocity
        self.headings[:,i] = state.heading
        self.angular_velocities[:,i] = state.angular_velocity
        self.steering_angles[:,i] = state.steering_angle
        self.control_throttle_positions[:,i] = control.throttle_position
        self.control_steering_angles[:,i] = control.steering_angle

import track

def world():
    trk = track.load_csv("3yp_track2500")
    # Scale the track (its current design is for full sized cars)
    trk = track.Track(matrix=trk.matrix * 0.04)

    t_max = 10
    dt = 0.01
    n = int(t_max / dt)
    t = 0

    # Arrays to store state and control inputs
    history = SimHistory.empty(n)

    state = VehicleState(position=(trk.x()[0], trk.y()[0]))

    control_model = controlmodel.SpenglerGammeterBicycle()
    controller = MPCPositionVelocityController(20, control_model, trk)

    control = ControllerOutput(throttle_position=0, steering_angle=0)

    vehicle_model = vehiclemodel.SpenglerGammeterBicycle()

    print("Starting simulation")

    start = timer()

    #try:
    for i in range(0, n):
        t = float(i) * dt

        if i % 100 == 0:
            print("Position", state.position, "Heading", state.heading)

        ctrl_start = timer()
        control = controller.step(dt, state.controller_input())
        ctrl_elapsed = timer() - ctrl_start

        if i % 100 == 0:
            print("Controller time", ctrl_elapsed)
            print("Control", control)

        state = vehicle_model.step(dt, state, control)
        
        # Record vehicle state and control inputs
        history.record(i, state, control)
    #except:
    #    print("Simulation failed after", t, "s")


    end = timer()
    print("Simulation of", t_max, "s took", end - start, "s")
    print("Plotting results")

    visualisation.show(trk, dt, history)

if __name__ == "__main__":
    world()

