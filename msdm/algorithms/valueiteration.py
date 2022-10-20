from dataclasses import dataclass
import numpy as np

from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp.policy.tabularpolicy_new import TabularPolicy
from msdm.core.mdp_tables import StateTable, StateActionTable
from msdm.core.algorithmclasses import Plans, PlanningResult

class ValueIteration(Plans):
    VALUE_DECIMAL_PRECISION = 10
    def __init__(
        self,
        max_iterations=int(1e5),
        max_residual=1e-5,
        _version="vectorized"
    ):
        self.max_iterations = max_iterations
        self.max_residual = max_residual
        self._version = _version

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        if self._version == 'dict':
            state_values, action_values, iterations = value_iteration_tabular(
                mdp,
                max_residual=self.max_residual,
                max_iterations=self.max_iterations
            )
            state_values = StateTable.from_dict(
                state_values=state_values
            )
            action_values = StateActionTable.from_dict(
                action_values=action_values,
                default_value=float('-inf')
            )
            policy = {}
            round_val = lambda v: round(v, self.VALUE_DECIMAL_PRECISION)
            for s in action_values.keys():
                maxq = max([round_val(v) for v in action_values[s].values()])
                max_actions = [a for a in mdp.actions(s) if round_val(action_values[s][a]) == maxq]
                if len(max_actions) == 0:
                    # dead ends are handled by turning them into a uniform distribution
                    # since they are all equally -inf
                    max_actions = mdp.action_list
                policy[s] = DictDistribution({a: 1/len(max_actions) for a in max_actions})
            policy = TabularPolicy.from_dict(
                action_values=policy,
                default_value=0
            )
        elif self._version == "vectorized":
            state_values, action_values, iterations = value_iteration_vectorized(
                transition_matrix=mdp.transition_matrix,
                terminal_state_vector=~mdp.nonterminal_state_vec.astype(bool),
                discount_rate=mdp.discount_rate,
                reward_matrix=mdp.reward_matrix,
                action_matrix=mdp.action_matrix,
                max_residual=self.max_residual,
                max_iterations=self.max_iterations
            )
            # terminal dead ends are handled by turning them into a uniform distribution
            # since they are all equally -inf, but non-terminal dead end states 
            # are allowed to have their single action taken
            policy_matrix = np.isclose(
                action_values,
                np.max(action_values, axis=-1, keepdims=True),
                atol=10**(-self.VALUE_DECIMAL_PRECISION),
                rtol=0
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
        else:
            raise ValueError
        return ValueIterationResult(
            iterations=iterations,
            state_value=state_values,
            action_value=action_values,
            converged=iterations < (self.max_iterations - 1),
            initial_value=sum([state_values[s]*p for s, p in mdp.initial_state_dist().items()]),
            policy=policy
        )

@dataclass
class ValueIterationResult(PlanningResult):
    iterations : int
    converged : bool
    state_value : dict
    initial_value : float
    action_value: dict
    policy : TabularPolicy

def value_iteration_vectorized(
    transition_matrix,
    terminal_state_vector,
    discount_rate,
    reward_matrix,
    action_matrix,
    max_residual=1e-5,
    v=None,
    max_iterations=int(1e5)
):
    """
    """
    n_states = transition_matrix.shape[0]
    action_penalty = np.log(action_matrix)
    if v is None:
        v = np.zeros(n_states)
    for i in range(max_iterations):
        q = np.einsum(
            "san,san->san",
            transition_matrix,
            reward_matrix + discount_rate*v[np.newaxis, np.newaxis, :]
        )
        q[np.isnan(q)] = 0
        q[terminal_state_vector] = 0
        q = q.sum(-1) + action_penalty
        nv = np.max(q, axis=-1)
        if np.isclose(v, nv, atol=max_residual, rtol=0).all():
            break
        v = nv
    return v, q, i

def value_iteration_tabular(
    mdp: TabularMarkovDecisionProcess,
    max_residual=1e-5,
    max_iterations=int(1e5)
):
    v = {s: 0 for s in mdp.state_list}
    for i in range(max_iterations):
        q = {}
        for s in mdp.state_list:
            q[s] = {}
            for a in mdp.actions(s):
                q[s][a] = 0
                if mdp.is_terminal(s):
                    continue
                for ns, prob in mdp.next_state_dist(s, a).items():
                    q[s][a] += prob*(mdp.reward(s, a, ns) + mdp.discount_rate*v[ns])
        residual = 0
        for s in mdp.state_list:
            if len(q[s]) > 0:
                new_value = max(q[s].values())
            else:
                new_value = float('-inf') #state is a dead end
            residual = max(residual, abs(v[s] - new_value))
            v[s] = new_value
        if residual < max_residual:
            break
    return v, q, i