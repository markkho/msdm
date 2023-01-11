from dataclasses import dataclass
import numpy as np
import warnings

from msdm.core.distributions import DictDistribution
from msdm.core.mdp import TabularMarkovDecisionProcess, TabularPolicy, \
    StateTable, StateActionTable
from msdm.core.algorithmclasses import Plans, PlanningResult

class ValueIteration(Plans):
    def __init__(
        self,
        max_iterations=int(1e5),
        max_residual=1e-5,
        undefined_value=0,
        _version="vectorized"
    ):
        self.max_iterations = max_iterations
        self.max_residual = max_residual
        self._version = _version
        self.undefined_value = undefined_value
    
    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        if mdp.dead_end_state_vec.any():
            warnings.warn("MDP contains states where no actions can be taken. This can have unanticipated effects.")
        if mdp._unable_to_reach_absorbing.any():
            warnings.warn("MDP contains states that never reach an absorbing state. " +\
                f"Values for these states will be set using self.undefined_value={self.undefined_value}"
            )
        if self._version == 'dict':
            return self._dict_plan_on(mdp)
        elif self._version == "vectorized":
            return self._vectorized_plan_on(mdp)
        else:
            raise ValueError

    def _vectorized_plan_on(self, mdp: TabularMarkovDecisionProcess):
        transition_matrix = mdp.transition_matrix.copy()
        transition_matrix[mdp._unable_to_reach_absorbing,] = 0
        transition_matrix[mdp.absorbing_state_vec,] = 0
        state_action_reward_matrix = mdp.state_action_reward_matrix.copy()
        state_action_reward_matrix[mdp._unable_to_reach_absorbing,] = 0
        state_action_reward_matrix[mdp.absorbing_state_vec,] = 0
        state_values, action_values, iterations = value_iteration_vectorized(
            transition_matrix=transition_matrix,
            discount_rate=mdp.discount_rate,
            state_action_reward_matrix=state_action_reward_matrix,
            action_matrix=mdp.action_matrix.astype(bool),
            max_residual=self.max_residual,
            max_iterations=self.max_iterations
        )
        state_values[mdp._unable_to_reach_absorbing,] = self.undefined_value
        action_values[mdp._unable_to_reach_absorbing,] = self.undefined_value
        policy_matrix = np.isclose(
            action_values,
            np.max(action_values, axis=-1, keepdims=True),
        )
        policy_matrix = policy_matrix/policy_matrix.sum(-1, keepdims=True)
        single_action_states = mdp.action_matrix.sum(-1) == 1
        policy_matrix[single_action_states] = mdp.action_matrix[single_action_states]
        policy=TabularPolicy.from_state_action_lists(
            state_list=mdp.state_list,
            action_list=mdp.action_list,
            data=policy_matrix
        )
        state_values=StateTable.from_state_list(
            state_list=mdp.state_list,
            data=state_values
        )
        action_values=StateActionTable.from_state_action_lists(
            state_list=mdp.state_list,
            action_list=mdp.action_list,
            data=action_values
        )
        return ValueIterationResult(
            iterations=iterations,
            state_value=state_values,
            action_value=action_values,
            converged=iterations < (self.max_iterations - 1),
            initial_value=sum([state_values[s]*p for s, p in mdp.initial_state_dist().items()]),
            policy=policy
        )
    
    def _dict_plan_on(self, mdp: TabularMarkovDecisionProcess):
        state_values, action_values, iterations = value_iteration_tabular(
            mdp,
            max_residual=self.max_residual,
            max_iterations=self.max_iterations
        )
        state_values = {
            s: v if not mdp._unable_to_reach_absorbing[mdp.state_list.index(s)]
            else self.undefined_value
            for s, v in state_values.items()
        }
        state_values = StateTable.from_dict(
            state_values=state_values
        )
        action_values = StateActionTable.from_dict(
            action_values=action_values,
            default_value=float('-inf')
        )
        policy = {}
        for s in action_values.keys():
            maxq = max(action_values[s].values())
            max_actions = [a for a in mdp.actions(s) if np.isclose(action_values[s][a], maxq)]
            if len(max_actions) == 0: #dead end
                max_actions = mdp.action_list
            policy[s] = DictDistribution({a: 1/len(max_actions) for a in max_actions})
        policy = TabularPolicy.from_dict(
            action_values=policy,
            default_value=0
        )
        return ValueIterationResult(
            iterations=iterations,
            state_value=state_values,
            action_value=action_values,
            converged=iterations < (self.max_iterations - 1),
            initial_value=sum([state_values[s]*p for s, p in mdp.initial_state_dist().items()]),
            policy=policy
        )

from msdm.core.table.tablemisc import dataclass_repr_html_MixIn
@dataclass
class ValueIterationResult(PlanningResult,dataclass_repr_html_MixIn):
    iterations : int
    converged : bool
    state_value : dict
    initial_value : float
    action_value: dict
    policy : TabularPolicy

def value_iteration_vectorized(
    transition_matrix,
    discount_rate,
    state_action_reward_matrix,
    action_matrix,
    state_values=None,
    max_residual=1e-5,
    max_iterations=int(1e5)
):
    """
    """
    n_states = transition_matrix.shape[0]
    action_penalty = np.log(action_matrix)
    if state_values is None:
        state_values = np.zeros(n_states)
    for i in range(max_iterations):
        future_action_values = \
            np.einsum("san,n->sa", transition_matrix, state_values)
        action_values = \
            state_action_reward_matrix +\
            discount_rate*future_action_values +\
            action_penalty
        next_state_values = np.max(action_values, axis=-1)
        if np.isclose(state_values, next_state_values, atol=max_residual, rtol=0).all():
            break
        state_values = next_state_values
    return state_values, action_values, i

def value_iteration_tabular(
    mdp: TabularMarkovDecisionProcess,
    max_residual=1e-5,
    max_iterations=int(1e5)
):
    state_values = {s: 0 for s in mdp.state_list}
    for i in range(max_iterations):
        action_values = {}
        for si, s in enumerate(mdp.state_list):
            action_values[s] = {}
            for a in mdp.actions(s):
                action_values[s][a] = 0
                if mdp.is_absorbing(s) or mdp._unable_to_reach_absorbing[si]:
                    continue
                for ns, prob in mdp.next_state_dist(s, a).items():
                    action_values[s][a] += prob*(mdp.reward(s, a, ns) + mdp.discount_rate*state_values[ns])
        residual = 0
        for s in mdp.state_list:
            if len(action_values[s]) > 0:
                new_value = max(action_values[s].values())
            else:
                new_value = float('-inf') #state is a dead end
            residual = max(residual, abs(state_values[s] - new_value))
            state_values[s] = new_value
        if residual < max_residual:
            break
    return state_values, action_values, i