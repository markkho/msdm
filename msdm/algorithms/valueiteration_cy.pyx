import cython
import numpy as np
INFINITY = cython.declare(cython.double, np.inf)

@cython.locals(
    transition_matrix=cython.double[:, :, :],
    absorbing_state_vec=cython.int[:],
    state_action_reward_matrix=cython.double[:, :],
    action_matrix=cython.int[:, :],
    discount_rate=cython.double,
    value_error_threshold=cython.double,
    max_iterations=cython.int,
    state_values=cython.double[:],
)
@cython.returns(cython.int)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def value_iteration(
    transition_matrix : np.ndarray,
    absorbing_state_vec : np.ndarray,
    state_action_reward_matrix : np.ndarray,
    action_matrix : np.ndarray,
    discount_rate : float,
    value_error_threshold : float,
    max_iterations : int,
    state_values : np.ndarray,
):
    cython.declare(s=cython.int, a=cython.int, ns=cython.int, i=cython.int)
    cython.declare(action_val=cython.double, max_aval=cython.double, max_diff=cython.double)
    n_states = cython.declare(cython.int, transition_matrix.shape[0])
    n_actions = cython.declare(cython.int, transition_matrix.shape[1])
    for i in range(max_iterations):
        max_diff = 0.
        for s in range(n_states):
            if absorbing_state_vec[s]:
                state_values[s] = 0
                continue
            max_aval = -INFINITY
            for a in range(n_actions):
                if action_matrix[s, a] == 0:
                    action_val = -INFINITY
                    continue
                action_val = state_action_reward_matrix[s, a]
                for ns in range(n_states):
                    if transition_matrix[s, a, ns] == 0:
                        continue
                    action_val += discount_rate*transition_matrix[s, a, ns]*state_values[ns]
                if action_val > max_aval:
                    max_aval = action_val
            max_diff = max(abs(max_aval - state_values[s]), max_diff)
            state_values[s] = max_aval
        if max_diff < value_error_threshold:
            break
    return i