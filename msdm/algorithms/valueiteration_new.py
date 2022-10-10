import numpy as np

from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy
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
            state_value, action_value, iterations = value_iteration_tabular(
                mdp,
                max_residual=self.max_residual,
                max_iterations=self.max_iterations
            )
        elif self._version == "vectorized":
            v, q, _, iterations = value_iteration_vectorized(
                transition_matrix=mdp.transition_matrix,
                terminal_state_vector=~mdp.nonterminal_state_vec.astype(bool),
                discount_rate=mdp.discount_rate,
                reward_matrix=mdp.reward_matrix,
                action_matrix=mdp.action_matrix,
                max_residual=self.max_residual,
                max_iterations=self.max_iterations
            )
            action_value = {}
            for s in mdp.state_list:
                si = mdp.state_index[s]
                action_value[s] = {}
                for a in mdp.actions(s):
                    ai = mdp.action_index[a]
                    action_value[s][a] = q[si][ai]
            state_value = dict(zip(mdp.state_list, v))
        else:
            raise ValueError
        
        policy = {}
        round_val = lambda v: round(v, self.VALUE_DECIMAL_PRECISION)
        for s in mdp.state_list:
            if len(action_value[s]) == 0:
                continue
            maxq = max([round_val(v) for v in action_value[s].values()])
            max_actions = [a for a in mdp.actions(s) if round_val(action_value[s][a]) == maxq]
            policy[s] = DictDistribution({a: 1/len(max_actions) for a in max_actions})
        return PlanningResult(
            iterations=iterations,
            converged=iterations < (self.max_iterations - 1),
            state_value=state_value,
            initial_value=sum([state_value[s]*p for s, p in mdp.initial_state_dist().items()]),
            action_value=action_value,
            policy=TabularPolicy(policy)
        )

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
    pi = q == np.max(q, axis=-1, keepdims=True)
    pi = pi/pi.sum(-1, keepdims=True)
    return v, q, pi, i

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