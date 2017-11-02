import numpy as np

from utils import stride

class SparseQPBuilder:
    def __init__(self, horizon_len, num_states, num_inputs, num_inequalities):
        self.horizon_len = horizon_len
        self.num_model_states = num_states
        self.num_model_inputs = num_inputs

        num_qp_inputs = horizon_len * num_inputs
        num_qp_states = horizon_len * num_states

        # delta_x = C * delta_u
        self.C = np.zeros((num_qp_states, num_qp_inputs))
        self.current_model_idx = 0

        self.Q = np.zeros((num_qp_states, num_qp_states))
        self.k = np.zeros(num_qp_states)
        self.R = np.zeros((num_qp_inputs, num_qp_inputs))

        self.x_0 = np.zeros(num_qp_states)
        self.u_0 = np.zeros(num_qp_inputs)

        self.A_ineq = np.zeros((num_inequalities, num_qp_inputs))
        self.B_ineq = np.zeros(num_inequalities)
        self.ineq_idx = 0

    def reset(self):
        self.current_model_idx = 0
        self.ineq_idx = 0

    def add_model(self, A, B, x_0, u_0):
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
        #for j in range(0, i):
        #    C[stride(i, ns, j, ni)] = A.dot(C[stride(i - 1, ns, j, ni)])
        if i > 0:
            C[stride(i, ns), 0:i * ni] = A.dot(C[stride(i - 1, ns), 0:i * ni])

        self.u_0[stride(i, ni)] = u_0
        self.x_0[stride(i, ns)] = x_0
        
        self.current_model_idx += 1
    
    # As in 3 * (x + 2)^2
    def set_offset_state_cost(self, horizon_idx, Q_stage, k):
        # ||x + k||Q^2
        # x'Qx + 2x'Qk (+ k'Qk)
        # given: x = x_0 + Cu
        # (x_0 + Cu)'Q(x_0 + Cu) + 2(x_0 + Cu)'Qk
        # u'C'QCu + 2u'C'Qx_0 (+ x_0'Qx_0) (+ 2x_0'Qk) + 2u'C'Qk
        # u'(C'QC)u + 2u'C'Q(x_0 + k)

        # This fills in the Q and k matrix
        ns = self.num_model_states

        assert Q_stage.shape == (ns, ns)
        assert k.shape == (ns,)

        self.Q[stride(horizon_idx, ns, horizon_idx, ns)] = Q_stage
        self.k[stride(horizon_idx, ns)] = k

    def set_input_gradient_cost(self, horizon_idx, R_grad):
        horizon = self.horizon_len
        ni = self.num_model_inputs

        assert R_grad.shape == (ni, ni)

        if horizon_idx == 0:
            self.R[stride(0, ni, 0, ni)] = R_grad
            self.R[stride(0, ni, 1, ni)] = -R_grad
        elif horizon_idx == horizon - 1:
            self.R[stride(horizon_idx, ni, horizon_idx - 1, ni)] = -R_grad
            self.R[stride(horizon_idx, ni, horizon_idx, ni)] = R_grad
        else:
            self.R[stride(horizon_idx, ni, horizon_idx - 1, ni)] = -R_grad
            self.R[stride(horizon_idx, ni, horizon_idx, ni)] = 2 * R_grad
            self.R[stride(horizon_idx, ni, horizon_idx + 1, ni)] = -R_grad

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

    def build_qp(self, x_k):
        C = self.C
        Q = self.Q
        R = self.R
        k = self.k

        CTQ = C.T.dot(Q)

        H = CTQ.dot(C) + R
        # Use delta states so no need to compute M
        # M is determined more accurately from the linearisation set-points
        f = CTQ.dot(self.x_0 + k) + R.dot(self.u_0)
        #f = CTQ.dot(M.dot(x_k) + k) + R * u_0

        return H, f, self.A_ineq, self.B_ineq

