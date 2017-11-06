from math import atan2
import numpy as np
from timeit import default_timer as timer

from . import Controller, ControllerInput, ControllerOutput, CondensedQPBuilder
from controlmodel import ControlModel
from track import Track, CurvilinearMapping, parameterise
from utils import stride, norm_angle

class MPCPositionVelocityController(Controller):
    def __init__(self, horizon: int, model: ControlModel, trk: Track):
        self.bootstrap = True
        self.horizon = horizon
        self.num_inputs = model.num_inputs()
        self.input_history = np.zeros(self.num_inputs * self.horizon)
        self.model = model
        self.trk = trk
        self.trk_mapping = parameterise(trk)

    def step(self, dt: float, state: ControllerInput) -> ControllerOutput:
        v_max = 1
        
        model = self.model
        x_k = model.state_from_controller_input(state)

        num_inputs = self.num_inputs
        num_states = model.num_states()

        num_inequality_constraints = self.horizon * 6

        builder = CondensedQPBuilder(self.horizon, num_states, num_inputs, num_inequality_constraints)
 
        u_0 = self.input_history[:]

        x_i = x_k

        start = timer()
        arc_distance = self.trk_mapping.theta(x_k[0], x_k[1])
        #print("KD lookup took", timer() - start)

        arc_distance += 2 * v_max * dt

        sim_time = 0
        disc_time = 0
        model_add_time = 0
        target_time = 0
        for i in range(0, self.horizon):
            u_i = u_0[stride(i, num_inputs)]
            
            start = timer()
            A, B = model.discretise(dt, x_i, u_i)
            disc_time += timer() - start
            #A = np.eye(3)
            #B = [[0.01, 0], [0, 0], [0, 0.01]]

            start = timer()
            x_i = model.step(dt, x_i, u_i)
            sim_time += timer() - start

            start = timer()
            builder.add_model(A, B, x_i, u_i)
            model_add_time += timer() - start

            # Choose target x and y values
            start = timer()
            arc_distance += v_max * dt
            x_target = self.trk_mapping.xy(arc_distance)
            dx_target = self.trk_mapping.xy(arc_distance, 1)
            phi_target = atan2(dx_target[1], dx_target[0])
            phi_target = np.unwrap([x_i[2], phi_target])[1]
            target_time += timer() - start
            #print("x", [x_i[0], x_i[1]], "x_target", x_target, "phi", x_i[2], "target", phi_target)

            # Penalise x and y deviations only
            Q = np.eye(num_states) * 0
            Q[0,0] = 20
            Q[1,1] = 20
            Q[2,2] = 10
            k = np.zeros(num_states)
            k[0] -= x_target[0]
            k[1] -= x_target[1]
            k[2] -= phi_target
            builder.set_offset_state_cost(i, Q, k)

            # Penalise changes in throttle position and steering angle
            R = np.eye(num_inputs) * 0
            R[0,0] = 0.5
            R[1,1] = 1
            builder.set_input_gradient_cost(i, R)

            # Input limits
            builder.set_input_bound(i, 0, 0, v_max)
            builder.set_input_delta_bound(i, 0, -0.5, 0.5)
            builder.set_input_delta_bound(i, 1, -1, 1)
        #print("Simulation", sim_time, "Discretisation", disc_time, "Model Add", model_add_time, "Target", target_time)

        start = timer()
        u = builder.solve_qp()
        #print("Sovler took", timer() - start)

        self.input_history[0:(self.horizon - 1) * num_inputs] = u[num_inputs:len(u)]
        self.input_history[stride(self.horizon - 1, num_inputs)] = u[stride(self.horizon - 2, num_inputs)]

        return model.control_to_controller_output(u[0:num_inputs])


