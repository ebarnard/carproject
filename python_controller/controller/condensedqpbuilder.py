import numpy as np
from quadprog import solve_qp

from utils import stride

class CondensedQPBuilder:
    def __init__(self, N, Q_stage, R_stage_grad, input_lb, input_ub):
        num_states = len(Q_stage)
        num_inputs = len(R_stage_grad)

        assert len(input_lb) == num_inputs
        assert len(input_ub) == num_inputs

        self.horizon_len = N
        self.num_model_states = num_states
        self.num_model_inputs = num_inputs
        
        num_qp_inputs = N * num_inputs
        num_qp_states = N * num_states

        # delta_x = C * delta_u
        self.C = np.zeros((num_qp_states, num_qp_inputs))
        self.current_model_idx = 0

        self.Q = CondensedQPBuilder.make_Q(N, Q_stage)
        self.k = np.zeros(num_qp_states)
        self.R = CondensedQPBuilder.make_R(N, R_stage_grad)

        self.x_0 = np.zeros(num_qp_states)
        self.u_0 = np.zeros(num_qp_inputs)

        # An upper and lower bound for each input
        self.ineq_idx = 0
        num_inequalities = 2 * num_qp_inputs
        self.A_ineq = np.zeros((num_inequalities, num_qp_inputs))
        self.B_ineq = np.zeros(num_inequalities)

        self.input_lb = input_lb
        self.input_ub = input_ub

    def make_R(horizon, R_stage_change):
        ni = len(R_stage_change)
        R_stage_change = np.diag(R_stage_change)

        assert R_stage_change.shape == (ni, ni)

        D = np.identity(horizon * ni)
        R = np.zeros((horizon * ni, horizon * ni))
        
        R[stride(0, ni, 0, ni)] = R_stage_change
        for i in range(1, horizon):
            R[stride(i, ni, i - 1, ni)] = -np.identity(ni)
            R[stride(i, ni, i, ni)] = R_stage_change

        return D.T * R * D

    def make_Q(horizon, Q_stage):
        ns = len(Q_stage)
        Q_stage = np.diag(Q_stage)

        Q = np.zeros((horizon * ns, horizon * ns))

        for i in range(0, horizon):
            Q[stride(i, ns, i, ns)] = Q_stage

        return Q

    def reset(self):
        self.current_model_idx = 0
        self.ineq_idx = 0

    def add_model(self, A, B, x_0, u_0, x_target):
        # Condensed model
        ns = self.num_model_states
        ni = self.num_model_inputs

        assert A.shape == (ns, ns)
        assert B.shape == (ns, ni)

        i = self.current_model_idx
        C = self.C

        if i == self.horizon_len:
            raise "Too many models added"

        C[stride(i, ns, i, ni)] = B
        if i > 0:
            C[stride(i, ns), 0:i * ni] = A.dot(C[stride(i - 1, ns), 0:i * ni])

        self.u_0[stride(i, ni)] = u_0
        self.x_0[stride(i, ns)] = x_0
        
        # Target state value
        # ||x + k||Q^2
        # x'Qx + 2x'Qk (+ k'Qk)
        # given: x = x_0 + Cu
        # (x_0 + Cu)'Q(x_0 + Cu) + 2(x_0 + Cu)'Qk
        # u'C'QCu + 2u'C'Qx_0 (+ x_0'Qx_0) (+ 2x_0'Qk) + 2u'C'Qk
        # u'(C'QC)u + 2u'C'Q(x_0 + k)
        assert x_target.shape == (ns,)
        self.k[stride(i, ns)] = x_target

        # Input bounds
        for i in range(0, ni):
            self.set_input_bound(self.current_model_idx, i, self.input_lb[i], self.input_ub[i])

        self.current_model_idx += 1
    
    def set_input_bound(self, horizon_idx, input_idx, minv, maxv):
        # Ax <= b
        ni = self.num_model_inputs
        u_0 = self.u_0[horizon_idx * ni + input_idx]
        self.set_input_delta_bound(horizon_idx, input_idx, minv - u_0, maxv - u_0)

    def set_input_delta_bound(self, horizon_idx, input_idx, minv, maxv):
        # Ax <= b
        i = self.ineq_idx
        ni = self.num_model_inputs

        # u <= k
        self.A_ineq[i, horizon_idx * ni + input_idx] = 1
        self.B_ineq[i] = maxv

        # u >= k
        # -u <= -k
        self.A_ineq[i + 1, horizon_idx * ni + input_idx] = -1
        self.B_ineq[i + 1] = -minv

        self.ineq_idx += 2

    def solve_qp(self):
        if self.current_model_idx != self.horizon_len:
            raise "Not all stages updated"

        C = self.C
        Q = self.Q
        R = self.R
        k = self.k

        CTQ = C.T.dot(Q)

        H = CTQ.dot(C) + R
        
        # Previous input is used to calculate the first input difference cost
        previous_input = np.zeros(self.num_model_inputs * self.horizon_len)
        previous_input[0:self.num_model_inputs] = self.u_0[0:self.num_model_inputs]

        f = CTQ.dot(self.x_0 - k) + R.dot(self.u_0) - previous_input

        A_in = self.A_ineq
        b_in = self.B_ineq

        delta_u,_,_,_,_,_ = solve_qp(H, -f, -A_in.T, -b_in, 0, False)

        self.reset()

        return delta_u + self.u_0

