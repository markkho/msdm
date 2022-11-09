from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

from msdm.core.mdp import TabularMarkovDecisionProcess, TabularPolicy, \
    StateActionTable, StateTable
from msdm.core.algorithmclasses import Plans, PlanningResult

class MultichainPolicyIteration(Plans):
    """
    Multichain average reward policy iteration algorithm
    as described in Puterman (1994) Section 9.2.2.
    
    This algorithm can handle problems where the discount rate is 
    <= 1. It returns a gain-bias optimal policy, the
    corresponding gain (the "average recurrent value function"),
    and the bias (the "transient value function") for states and actions.
    """
    VALUE_DECIMAL_PRECISION = 10
    def __init__(
        self,
        max_iterations=int(1e5),
    ):
        self.max_iterations = max_iterations

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        results = multichain_policy_iteration_vectorized(
            transition_matrix=mdp.transition_matrix,
            absorbing_state_vec=mdp.absorbing_state_vec.astype(bool),
            discount_rate=mdp.discount_rate,
            reward_matrix=mdp.reward_matrix,
            action_matrix=mdp.action_matrix.astype(bool),
            max_iterations=self.max_iterations
        )
        state_gain, action_gain, state_bias, action_bias, _, iterations = results
        gain_max_actions = np.isclose(
            action_gain, action_gain.max(-1, keepdims=True),
            atol=10**(-self.VALUE_DECIMAL_PRECISION),
            rtol=0
        )
        bias_max_actions = np.isclose(
            action_bias, action_bias.max(-1, keepdims=True),
            atol=10**(-self.VALUE_DECIMAL_PRECISION),
            rtol=0
        )
        policy_matrix = gain_max_actions & bias_max_actions
        policy_matrix = policy_matrix/policy_matrix.sum(-1, keepdims=True)
        policy=TabularPolicy.from_state_action_lists(
            state_list=mdp.state_list,
            action_list=mdp.action_list,
            data=policy_matrix
        )
        state_gain=StateTable.from_state_list(
            state_list=mdp.state_list,
            data=state_gain
        )
        action_gain=StateActionTable.from_state_action_lists(
            state_list=mdp.state_list,
            action_list=mdp.action_list,
            data=action_gain
        )
        state_bias=StateTable.from_state_list(
            state_list=mdp.state_list,
            data=state_bias
        )
        action_bias=StateActionTable.from_state_action_lists(
            state_list=mdp.state_list,
            action_list=mdp.action_list,
            data=action_bias
        )
        return MultichainPolicyIterationResult(
            iterations=iterations,
            state_gain=state_gain,
            action_gain=action_gain,
            initial_gain=sum(state_gain[s]*p for s, p in mdp.initial_state_dist().items()),
            state_value=state_bias,
            action_value=action_bias,
            initial_value=sum(state_bias[s]*p for s, p in mdp.initial_state_dist().items()),
            converged=iterations < (self.max_iterations - 1),
            policy=policy
        )

@dataclass
class MultichainPolicyIterationResult(PlanningResult):
    iterations : int
    converged : bool
    state_gain : dict
    initial_gain : float
    action_gain : dict
    state_value: dict
    initial_value: float
    action_value : dict
    policy : TabularPolicy

def multichain_policy_iteration_vectorized(
    transition_matrix,
    discount_rate,
    reward_matrix,
    absorbing_state_vec,
    action_matrix,
    max_iterations,
    policy=None
):
    assert absorbing_state_vec.dtype == bool
    assert action_matrix.dtype == bool
    assert action_matrix[~absorbing_state_vec].any(-1).all(), \
        "MDP has non-terminal states where no action can be taken (deadends)"
    
    if policy is None:
        policy = np.argmax(action_matrix, axis=-1)
    
    sa_rf = np.einsum(
        "san,san->sa", 
        transition_matrix,
        reward_matrix
    )
    sa_rf[absorbing_state_vec] = 0
    action_penalty = np.log(action_matrix)
    n_states = transition_matrix.shape[0]
    ss_range = np.arange(len(transition_matrix))
    eye = np.eye(n_states)
    zeros = np.zeros(eye.shape)
        
    for i in range(max_iterations):
        # Policy Evaluation
        # Construct a markov chain and analyze it to get the
        # (non-absorbing) recurrent classes.
        # If there are no recurrent classes, we don't need to augment
        # the system of linear equations
        s_rf = sa_rf[ss_range, policy]
        mp = discount_rate*transition_matrix[ss_range, policy]
        mp[absorbing_state_vec] = 0
        if discount_rate < 1.0:
            ref_eq = []
        else:
            reachable = floyd_warshall(mp) < float('inf')
            transient = (reachable & ~reachable.T).any(-1)
            nonrecurrent_states = transient | ~np.isclose(mp.sum(-1), 1)
            communicating = reachable & reachable.T
            communicating[nonrecurrent_states,:] = False
            communicating[:,nonrecurrent_states] = False
            rec_classes = np.unique(communicating, axis=0)
            rec_classes = rec_classes[rec_classes.any(-1)]
            if rec_classes.any():
                ref_vals = np.argmax(rec_classes, axis=-1)
                ref_eq = np.zeros((len(ref_vals), n_states*2))
                ref_eq[np.arange(len(ref_vals)), ref_vals + n_states] = 1
                ref_eq = [[ref_eq]]
            else:
                ref_eq = []

        # evaluate policy by solving system of linear
        # equations to get gain and bias vectors
        # note we need to set the bias for a state in each
        # recurrent class to 0
        coeff_block = np.block([
            [mp - eye, zeros],
            [-eye, mp - eye],
            *ref_eq
        ])
        ind_gain_rows = independent_row_indices(coeff_block[:n_states])
        ind_rows = ind_gain_rows + list(range(n_states, len(coeff_block)))
        gram_rf = np.zeros(coeff_block.shape[0])
        gram_rf[n_states:2*n_states] = -s_rf
        coeff_block = coeff_block[ind_rows,]
        gram_rf = gram_rf[ind_rows]

        # calculate minimum norm solution
        gram_matrix = coeff_block@coeff_block.T
        gram_solution = np.linalg.solve(
            gram_matrix,
            gram_rf
        ) 
        gain_bias = coeff_block.T@gram_solution
        gain, bias = gain_bias[:n_states], gain_bias[n_states:]

        # Policy improvement based on *gain*
        # Note: We always break max ties on the side of the previous policy
        gain_q = np.einsum("san,n->sa", transition_matrix, gain) + action_penalty
        new_policy = np.argmax(gain_q, axis=-1)
        policy_is_max = np.isclose(gain_q[ss_range, policy], gain_q[ss_range, new_policy])
        new_policy[policy_is_max] = policy[policy_is_max]
        if (new_policy != policy).any():
            policy = new_policy
            continue

        # Policy improvement based on *bias*
        # Note: We always break max ties on the side of the previous policy
        bias_q = sa_rf + discount_rate*np.einsum("san,n->sa", transition_matrix, bias) + action_penalty
        bias_q[absorbing_state_vec] = 0
        new_policy = np.argmax(bias_q, axis=-1)
        policy_is_max = np.isclose(bias_q[ss_range, policy], bias_q[ss_range, new_policy])
        new_policy[policy_is_max] = policy[policy_is_max]
        if (new_policy == policy).all():
            break
        policy = new_policy
    return gain, gain_q, bias, bias_q, policy, i

def independent_row_indices(mat):
    ind_vecs = []
    for i in range(mat.shape[0]):
        submat = mat[ind_vecs + [i,], ]
        if not np.isclose(np.linalg.det(submat@submat.T), 0):
            ind_vecs.append(i)
    return ind_vecs