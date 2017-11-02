from math import pi, sin, cos, atan, atan2
import numpy as np
from scipy.integrate import ode
import typing

from controller import ControllerOutput
from utils import norm_angle
from vehiclemodel import VehicleModel, VehicleState

class Params(typing.NamedTuple):
    mass: float
    I_z: float
    com_to_front: float
    com_to_rear: float
    reduced_drag_coefficient: float

def default_params() -> Params:
    com_to_front=0.04
    com_to_rear=0.06
    mass = 0.1
    # Ellipsoid I_z
    I_z = mass * (0.05*0.05) * 2 / 5
    reduced_drag_coefficient = 0.000576

    return Params(mass=mass, I_z=I_z, com_to_front=com_to_front, com_to_rear=com_to_rear, reduced_drag_coefficient=reduced_drag_coefficient)

class PacejkaBicycleModel(VehicleModel):
    def __init__(self):
        self.params = default_params()

    def state_from_vehicle_state(self, state: VehicleState):
        X, Y = state.position
        v_X, v_Y = state.velocity

        # Vehicle angle in global coordinates
        phi = state.heading

        # Angular velocity of the vehicle
        omega = state.angular_velocity

        return [X, Y, phi, v_X, v_Y, omega, state.steering_angle, state.steering_angular_velocity]

    def state_to_vehicle_state(self, state) -> VehicleState:
        X_dt, Y_dt, phi_dt, v_X_dt, v_Y_dt, omega_dt, delta_front, delta_front_dot = state

        # Normalize heading to be between -pi and pi
        phi_dt = norm_angle(phi_dt)

        return VehicleState(position=(X_dt, Y_dt), velocity=(v_X_dt, v_Y_dt), heading=phi_dt, angular_velocity=omega_dt, steering_angle = delta_front, steering_angular_velocity = delta_front_dot)

    def state_equation(t: float, y, control: ControllerOutput):
        params = self.params

        # See above for variable descriptions
        X, Y, phi, v_X, v_Y, omega, delta_front, delta_front_dot = y

        # Velocity along the vehicle line
        v_x = v_X * cos(phi) + v_Y * sin(phi)

        # Lateral velocity
        v_y = -v_X * sin(phi) + v_Y * cos(phi)

        # Vehicle mass
        mass = params.mass

        # Z
        I_z = params.I_z

        l_front = params.com_to_front
        l_rear = params.com_to_rear

        # Convert throttle position to torque/force
        tyre_force = control.throttle_position * 0.01

        # Front wheel driving force (at angle delta_front + alpha_front)
        # car is rear wheel powered so this is zero
        f_front_x = 0

        # Rear wheel driving force (at angle alpha_rear)
        f_rear_x = tyre_force

        # Front wheel steering dynamics
        # Constants from LQR - Q unit, R = 0.01
        delta_front_dot_dot = (control.steering_angle - delta_front) * 10 - delta_front_dot * 4.4721
        
        # Front wheel slip angle anticlockwise from vehicle centerline
        # Points in the direction the car is travelling, not the direction the wheels are pointing
        # Includes omega term as vehicle can be rotating about its centre of mass
        alpha_front = -atan2(l_front * omega + v_y, v_x)

        # Rear wheel slip angle anticlockwise from vehicle centerline
        alpha_rear = -atan2(-l_rear * omega + v_y, v_x)

        # Front wheel lateral friction force (90deg anticlockwise from the above)
        f_front_y = pacejka_slip_force(delta_front + alpha_front, params)

        # Rear wheel lateral force (90deg anticlockwise from the above)
        f_rear_y = pacejka_slip_force(alpha_rear, params)

        # Drag is assumed to only be significant in the forward direction
        f_drag = params.reduced_drag_coefficient * v_x * v_x

        # Acceleration along the line of the vehicle
        a_x = (f_rear_x - f_front_y * sin(delta_front) - f_drag) / mass + v_y * omega

        # Lateral acceleration
        a_y = (f_rear_y + f_front_y * cos(delta_front)) / mass - v_x * omega

        # Angular acceleration (anticlockwise)
        omega_dot = (f_front_y * l_front * cos(delta_front) - f_rear_y * l_rear) / I_z
        
        #print("start")
        #print(alpha_front)
        #print(alpha_rear)
        #print(a_x)
        #print(a_y)
        
        # Rotate acceleration vectors to global coordinate frame
        a_X = a_x * cos(phi) - a_y * sin(phi)
        a_Y = a_x * sin(phi) + a_y * cos(phi)

        #print(a_X)
        #print(a_Y)
        #if t > 0.001: raise
        y_dot = [v_X, v_Y, omega, a_X, a_Y, omega_dot, delta_front_dot, delta_front_dot_dot]

        return y_dot

def pacejka_slip_force(slip_angle: float, params: Params) -> float:
    # Numbers from simplified magic formula coefs for dry tarmac:
    # http://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
    B = 10
    C = 1.3
    D = 1
    E = 0.97
    F_z = 9.81 * params.mass

    # Slip factor
    # 0 if there is no slip and 1 if the car is travelling perpendicular to the wheel direction
    k = slip_angle * 2 / pi

    phi = (1 - E) * k + (E / B) * atan(B * k)
    return F_z * D * sin(C * atan(B * phi))
