import cython
import numpy as np
INFINITY = cython.declare(cython.double, np.inf)

@cython.locals(
    transition_matrix=cython.double[:, :, :],
    absorbing_state_vec=cython.int[:],
    state_action_reward_matrix=cython.double[:, :],
    action_matrix=cython.int[:, :],
    discount_rate=cython.double,
    max_iterations=cython.int,
    policy_evaluation_error_threshold=cython.double,
    init_evaluation_iterations=cython.int,
    loop_evaluation_iterations=cython.int,
    last_evaluation_iterations=cython.int,
    policy=cython.int[:],
    state_values=cython.double[:],
)
@cython.returns((cython.int, cython.int))
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def policy_iteration(
    transition_matrix : np.ndarray,
    absorbing_state_vec : np.ndarray,
    state_action_reward_matrix : np.ndarray,
    action_matrix : np.ndarray,
    discount_rate : float,
    max_iterations : int,
    policy_evaluation_error_threshold : float,
    init_evaluation_iterations : int,
    loop_evaluation_iterations : int,
    last_evaluation_iterations : int,
    policy : np.ndarray,
    state_values : np.ndarray,
):
    cython.declare(
        max_evaluation_i=cython.int, policy_change=cython.int, i=cython.int,
        use_random_policy=cython.int, evaluation_iterations=cython.int,
        evaluation_i=cython.int
    )
    max_evaluation_i = -1
    policy_change = True
    for i in range(max_iterations):
        use_random_policy = 0
        if i == 0:
            evaluation_iterations = init_evaluation_iterations
        elif not policy_change:
            evaluation_iterations = last_evaluation_iterations
        else:
            evaluation_iterations = loop_evaluation_iterations
        evaluation_i = policy_evaluation(
            transition_matrix,
            absorbing_state_vec,
            state_action_reward_matrix,
            action_matrix,
            discount_rate,
            policy_evaluation_error_threshold,
            evaluation_iterations,
            use_random_policy,
            policy,
            state_values,
        )
        max_evaluation_i = max(max_evaluation_i, evaluation_i)
        if i > 0 and not policy_change:
            break
        policy_change = policy_improvement(
            transition_matrix,
            absorbing_state_vec,
            state_action_reward_matrix,
            action_matrix,
            discount_rate,
            policy,
            state_values
        )
    return (i, max_evaluation_i)

@cython.cfunc
@cython.locals(
    transition_matrix=cython.double[:, :, :],
    absorbing_state_vec=cython.int[:],
    state_action_reward_matrix=cython.double[:, :],
    action_matrix=cython.int[:, :],
    discount_rate=cython.double,
    value_error_threshold=cython.double,
    max_iterations=cython.int,
    use_random_policy=cython.int,
    policy=cython.int[:],
    state_values=cython.double[:],
)
@cython.returns(cython.int)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def policy_evaluation(
    transition_matrix : np.ndarray,
    absorbing_state_vec : np.ndarray,
    state_action_reward_matrix : np.ndarray,
    action_matrix : np.ndarray,
    discount_rate : float,
    value_error_threshold : float,
    max_iterations : int,
    use_random_policy : int,
    policy : np.ndarray,
    state_values : np.ndarray,
):
    cython.declare(
        n_states=cython.int, n_actions=cython.int, i=cython.int,
        max_diff=cython.double, s=cython.int, a=cython.int, ns=cython.int,
        action_val=cython.double, n_available_actions=cython.int,
    )
    n_states = transition_matrix.shape[0]
    n_actions = transition_matrix.shape[1]
    for i in range(max_iterations):
        max_diff = 0.
        for s in range(n_states):
            if absorbing_state_vec[s]:
                state_values[s] = 0
                continue
            if use_random_policy:
                action_val = 0.
                n_available_actions = n_actions
                for a in range(n_actions):
                    if action_matrix[s, a] == 0:
                        n_available_actions -= 1
                        continue
                    action_val += state_action_reward_matrix[s, a]
                    for ns in range(n_states):
                        if transition_matrix[s, a, ns] == 0:
                            continue
                        action_val += discount_rate*transition_matrix[s, a, ns]*state_values[ns]
                if n_available_actions > 0:
                    action_val = action_val/n_available_actions
                else:
                    action_val = -INFINITY
            else:
                action_val = state_action_reward_matrix[s, policy[s]]
                for ns in range(n_states):
                    if transition_matrix[s, policy[s], ns] == 0:
                        continue
                    action_val += discount_rate*transition_matrix[s, policy[s], ns]*state_values[ns]
            max_diff = max(abs(action_val - state_values[s]), max_diff)
            state_values[s] = action_val
        if max_diff < value_error_threshold:
            break
    return i

@cython.cfunc
@cython.locals(
    transition_matrix=cython.double[:, :, :],
    absorbing_state_vec=cython.int[:],
    state_action_reward_matrix=cython.double[:, :],
    action_matrix=cython.int[:, :],
    discount_rate=cython.double,
    policy=cython.int[:],
    state_values=cython.double[:],
)
@cython.returns(cython.int)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def policy_improvement(
    transition_matrix : np.ndarray,
    absorbing_state_vec : np.ndarray,
    state_action_reward_matrix : np.ndarray,
    action_matrix : np.ndarray,
    discount_rate : float,
    policy : np.ndarray,
    state_values : np.ndarray,
):
    cython.declare(
        n_states=cython.int, n_actions=cython.int, s=cython.int,
        max_aval=cython.double, max_a=cython.int, a=cython.int, ns=cython.int,
        action_val=cython.double, policy_change=cython.int,
    )
    n_states = transition_matrix.shape[0]
    n_actions = transition_matrix.shape[1]
    policy_change = False
    for s in range(n_states):
        if absorbing_state_vec[s]:
            continue
        max_aval = -INFINITY
        max_a = policy[s]
        for a in range(n_actions):
            if action_matrix[s, a] == 0:
                continue
            action_val = state_action_reward_matrix[s, a]
            for ns in range(n_states):
                if transition_matrix[s, a, ns] == 0:
                    continue
                action_val += discount_rate*transition_matrix[s, a, ns]*state_values[ns]
            if action_val > max_aval:
                max_aval = action_val
                max_a = a
        if max_a != policy[s]:
            policy_change = True
            policy[s] = max_a
    return policy_change