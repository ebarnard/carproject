import numpy as np
import osqp
import scipy.sparse as sparse

from utils import stride

class OSQPBuilder:
    def __init__(self, N, Q_stage, R_stage_grad, input_lb, input_ub):
        ns = len(Q_stage)
        ni = len(R_stage_grad)

        assert len(input_lb) == ni
        assert len(input_ub) == ni

        self.horizon_len = N
        self.num_model_states = ns
        self.num_model_inputs = ni

        self.current_model_idx = 0

        self.build_A()

        self.Q_stage = Q_stage
        self.R_stage_grad = R_stage_grad

        self.x_0 = np.zeros(N * ns)
        self.u_0 = np.zeros(N * ni)
        self.x_target = np.zeros(N * ns)

        self.input_lb = input_lb
        self.input_ub = input_ub

        Q = sparse.diags(np.tile(Q_stage, N))

        # Penalty on ||Du + k||R^2
        # D is a gradient extraction matrix (1 diagonal + -1 subdiagonal)
        # k is 0 except for the first entry where it is u_k_-1 (i.e. the previous input)
        # u'D'RDu + 2u'D'Rk (+ k'Rk)
        # u = u_0 + u
        # (u_0 + u)'D'RD(u_0 + u) + 2(u_0 + u)'D'Rk
        # u'D'RDu + 2u'D'RDu_0 + 2u'D'Rk
        # u'D'RDu + 2uD'R(Du0 + k)
        D = sparse.kron(sparse.diags([1, -1], [0, -1], shape=(N, N)), sparse.eye(ni))
        R_grad = sparse.diags(np.tile(R_stage_grad, N))
        self.R = D.transpose() * R_grad * D

        self.P = sparse.block_diag((Q, self.R)).tocsc()

        self.l = np.zeros((ns + ni) * N)
        self.u = np.zeros((ns + ni) * N)
        self.k = np.zeros((ns + ni) * N)

        self.osqp = None
        self.results = None
        self.previous_input = np.zeros(ni)

    def build_A(self):
        ns = self.num_model_states
        ni = self.num_model_inputs
        N = self.horizon_len

        # Use ones to fix the sparsity structure as explicit zeros are sometimes eliminated
        A_stage = np.ones((ns, ns))
        B_stage = np.ones((ns, ni))

        # State evolution
        Ax = -sparse.eye(ns * N) + sparse.kron(sparse.diags(np.ones(N - 1), -1), A_stage)
        Au = sparse.kron(sparse.eye(N), B_stage)
        A = sparse.hstack([Ax, Au])

        # Input constraints
        Ab = sparse.hstack([sparse.csc_matrix((ni * N, ns * N)), sparse.eye(ni * N)])
        A = sparse.vstack([A, Ab])

        A = A.tocsc()
        A.sort_indices()

        # The indices into the sparse A matrix data vector corresponding to each cell of
        # the A_i and B_i matrices
        A_indices = [];
        B_indices = [];

        for i in range(0, N):
            B_indices.append(csc_index_matrix(A, stride(i, ns), slice(N * ns + i * ni, N * ns + (i + 1) * ni)))
            if i > 0:
                A_indices.append(csc_index_matrix(A, stride(i, ns), stride(i - 1, ns)))

        self.A = A
        self.A_nnz = self.A.nnz
        self.A_indices = A_indices
        self.B_indices = B_indices

    def add_model(self, A, B, x_0, u_0, x_target):
        ns = self.num_model_states
        ni = self.num_model_inputs

        assert A.shape == (ns, ns)
        assert B.shape == (ns, ni)
        assert len(x_0) == ns
        assert len(u_0) == ni
        assert len(x_target) == ns

        i = self.current_model_idx
        if i == self.horizon_len:
            raise "Too many models added"

        csc_assign_by_index(self.A, self.B_indices[i], B)
        if i > 0:
            csc_assign_by_index(self.A, self.A_indices[i - 1], A)

        self.u_0[stride(i, ni)] = u_0
        self.x_0[stride(i, ns)] = x_0
        self.x_target[stride(i, ns)] = x_target
        
        self.current_model_idx += 1

    def solve_qp(self):
        ns = self.num_model_states
        ni = self.num_model_inputs
        N = self.horizon_len

        # Input and state bounds
        self.l[N * ns:] = -self.u_0 + np.tile(self.input_lb, N)
        self.u[N * ns:] = -self.u_0 + np.tile(self.input_ub, N)

        # State penalties
        delta_x_target = self.x_0 - self.x_target
        qx = np.zeros(N * ns)
        for i in range(0, N):
            qx[i * ns:(i + 1) * ns] = self.Q_stage * delta_x_target[i * ns:(i + 1) * ns]

        # Input penalties
        qu = self.R.dot(self.u_0)
        # The previous value of u is not a variable but is needed to calculate the u difference 
        # penalty.
        qu[0:ni] -= self.R_stage_grad * self.previous_input
        
        q = np.concatenate((qx, qu))
        #print(self.A.toarray())
        #print(self.P.toarray())
        #print(q)
        #print(self.u_0)
        #print(self.l)
        #print(self.u)

        # Ensure that the sparsity structure of has (probably) not changed. This is an easy way to
        # accidentally break the solver.
        assert self.A.nnz == self.A_nnz
        
        if self.osqp is None:
            self.osqp = osqp.OSQP()
            self.osqp.setup(P=self.P, q=q, A=self.A, l=self.l, u=self.u, polish=False, verbose=True, eps_abs=1e-2)
        else:
            self.osqp.update(q=q, Ax=self.A.data, l=self.l, u=self.u)
            #TODO: Figure out why this makes the solver slower
            #x = self.results.x
            #y = self.results.y
            #x[0:(N - 1) * ns] = x[ns:(N) * ns]
            #x[N * ns:N * ns + (N - 1) * ni] = x[N * ns + ni:]
            #x_rot = np.concatenate((x[ns:N * ns], x[(N - 1) * ns:N * ns], x[N * ns + ni:], np.zeros(ni)))
            #y_rot = y#np.concatenate((y[ns:N * ns], y[(N - 1) * ns:N * ns], y[N * ns + ni:], np.zeros(ni)))
            #self.osqp.warm_start(x=x)

        self.results = self.osqp.solve()

        status_val = self.results.info.status_val
        if status_val == 1:
            pass
        elif status_val == -2:
            print("Max iterations")
        elif status_val == -5:
            raise KeyboardInterrupt()
        else:
            print("Solver error:", self.results.info.status)
            raise "Solver error"

        u = self.results.x[ns * N:] + self.u_0
        # Record the previous input for u difference penalty calculation.
        self.previous_input = u[0:ni]
        
        # Reset the builder to start from the first model
        self.soft_reset()

        return u

    def soft_reset(self):
        self.current_model_idx = 0

    def reset(self):
        self.soft_reset()
        self.osqp = None

        
def csc_index_matrix(A, r_slice, c_slice):
    if not sparse.isspmatrix_csc(A):
        raise "A must be a sparse CSC matrix"
    
    I = np.zeros(A[r_slice,c_slice].shape, dtype=np.int32)

    for c in range(c_slice.start, c_slice.stop):
        row_start = A.indptr[c]
        row_indices = list(A.indices[row_start:A.indptr[c + 1]])
        for r in range(r_slice.start, r_slice.stop):
            row_data_index = row_indices.index(r)
            data_index = row_start + row_data_index
            I[r - r_slice.start, c - c_slice.start] = data_index

    return I

def csc_assign_by_index(A, I, B):
    if not sparse.isspmatrix_csc(A):
        raise "A must be a sparse CSC matrix"

    A.data[I.ravel()] = B.ravel()