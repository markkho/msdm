from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

class ValueIteration(Plans):
    VALUE_DECIMAL_PRECISION = 10
    def __init__(
        self,
        max_iterations=int(1e5),
        max_residual=1e-5,
        _version="dict"
    ):
        self.max_iterations = max_iterations
        self.max_residual = max_residual
        self._version = _version

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        if self._version == 'dict':
            v, q, i = value_iteration_tabular(
                mdp,
                max_residual=self.max_residual,
                max_iterations=self.max_iterations
            )
            policy = {}
            round_val = lambda v: round(v, self.VALUE_DECIMAL_PRECISION)
            for s in mdp.state_list:
                if len(q[s]) == 0:
                    continue
                maxq = max([round_val(v) for v in q[s].values()])
                max_actions = [a for a in mdp.actions(s) if round_val(q[s][a]) == maxq]
                policy[s] = DictDistribution({a: 1/len(max_actions) for a in max_actions})
            return PlanningResult(
                iterations=i,
                converged=i < (self.max_iterations - 1),
                state_value=v,
                action_value=q,
                policy=TabularPolicy(policy)
            )
        else:
            raise ValueError

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