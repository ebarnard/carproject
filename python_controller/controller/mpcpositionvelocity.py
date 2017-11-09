from math import atan2
import numpy as np
from timeit import default_timer as timer

from . import Controller, ControllerInput, ControllerOutput, CondensedQPBuilder, OSQPBuilder
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

        # Penalise x and y deviations only
        Q_stage = np.concatenate(([20, 20, 3], np.zeros(self.model.num_states() - 3)))

        # Penalise changes in throttle position and steering angle
        R_stage_change = [0.1, 1]
        input_lb = [0, -1]
        input_ub = [1, 1]

        self.builder = OSQPBuilder(self.horizon, Q_stage, R_stage_change, input_lb, input_ub)
        #self.builder = CondensedQPBuilder(self.horizon, Q_stage, R_stage_change, input_lb, input_ub)

    def step(self, dt: float, state: ControllerInput) -> ControllerOutput:
        v_target = 2
        
        model = self.model
        
        num_inputs = self.num_inputs
        num_states = model.num_states()
        
        u_0 = self.input_history[:]

        x_i = model.state_from_controller_input(state)

        #start = timer()
        arc_distance = self.trk_mapping.theta(x_i[0], x_i[1])
        #print("KD lookup took", timer() - start)

        arc_distance += 2 * v_target * dt

        sim_time = 0
        disc_time = 0
        model_add_time = 0
        target_time = 0
        for i in range(0, self.horizon):
            u_i = u_0[stride(i, num_inputs)]
            
            #start = timer()
            A, B = model.discretise(dt, x_i, u_i)
            #disc_time += timer() - start

            #start = timer()
            x_i = model.step(dt, x_i, u_i)
            #sim_time += timer() - start

            # Choose target x and y values
            #start = timer()
            arc_distance += v_target * dt
            x_target = self.trk_mapping.xy(arc_distance)
            dx_target = self.trk_mapping.xy(arc_distance, 1)
            phi_target = atan2(dx_target[1], dx_target[0])
            phi_target = np.unwrap([x_i[2], phi_target])[1]
            #target_time += timer() - start

            k = np.zeros(num_states)
            k[0] = x_target[0]
            k[1] = x_target[1]
            k[2] = phi_target

            #start = timer()
            self.builder.add_model(A, B, x_i, u_i, k)
            #model_add_time += timer() - start

        start = timer()
        u = self.builder.solve_qp()
        #print("Simulation", sim_time, "Discretisation", disc_time, "Model Add", model_add_time, "Target", target_time, "Solve", timer() - start)

        self.input_history[0:(self.horizon - 1) * num_inputs] = u[num_inputs:len(u)]
        self.input_history[stride(self.horizon - 1, num_inputs)] = u[stride(self.horizon - 2, num_inputs)]

        return model.control_to_controller_output(u[0:num_inputs])


