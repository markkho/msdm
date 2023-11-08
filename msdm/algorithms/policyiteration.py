import pyximport
pyximport.install(language_level=3)
from msdm.algorithms.policyiteration_cy import policy_iteration
from dataclasses import dataclass
import numpy as np

from msdm.core.mdp import TabularMarkovDecisionProcess, TabularPolicy, \
    StateActionTable, StateTable
from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.exceptions import AlgorithmException

class PolicyIteration(Plans):
    def __init__(
        self,
        max_iterations: int = int(1e5),
        allow_no_discounting: bool = False,
        _policy_evaluation_error_threshold: float = 1e-10,
        _init_evaluation_iterations: int = 10,
        _loop_evaluation_iterations: int = 10,
        _last_evaluation_iterations: int = 1000,
    ) -> None:
        self.max_iterations = max_iterations
        self.allow_no_discounting = allow_no_discounting
        self.policy_evaluation_error_threshold = _policy_evaluation_error_threshold
        self.init_evaluation_iterations = _init_evaluation_iterations
        self.loop_evaluation_iterations = _loop_evaluation_iterations
        self.last_evaluation_iterations = _last_evaluation_iterations

    def plan_on(self, mdp: TabularMarkovDecisionProcess) -> "PolicyIterationResult":
        if mdp.discount_rate >= 1.0 and not self.allow_no_discounting:
            raise AlgorithmException(f"MDP has discount rate of {mdp.discount_rate}, but allow_no_discounting is False.")

        mdp.transition_matrix.setflags(write=True)
        mdp.state_action_reward_matrix.setflags(write=True)

        state_values = np.zeros(len(mdp.state_list), dtype=float)
        policy = np.zeros(len(mdp.state_list), dtype=np.intc)

        iterations, _ = policy_iteration(
            transition_matrix=mdp.transition_matrix,
            absorbing_state_vec=mdp.absorbing_state_vec.astype(np.intc),
            state_action_reward_matrix=mdp.state_action_reward_matrix,
            action_matrix=mdp.action_matrix.astype(np.intc),
            discount_rate=mdp.discount_rate,
            max_iterations=self.max_iterations,
            policy_evaluation_error_threshold=self.policy_evaluation_error_threshold,
            init_evaluation_iterations=self.init_evaluation_iterations,
            loop_evaluation_iterations=self.loop_evaluation_iterations,
            last_evaluation_iterations=self.last_evaluation_iterations,
            policy=policy,
            state_values=state_values
        )

        mdp.transition_matrix.setflags(write=False)
        mdp.state_action_reward_matrix.setflags(write=False)

        state_value = StateTable.from_state_list(mdp.state_list, state_values)
        action_values = \
            mdp.state_action_reward_matrix + \
            mdp.discount_rate * mdp.transition_matrix @ state_values
        action_values[mdp.absorbing_state_vec, :] = 0
        action_values[mdp.action_matrix == 0] = -np.inf
        action_value = StateActionTable.from_state_action_lists(
            mdp.state_list, mdp.action_list, action_values
        )
        
        policy = np.isclose(action_values, action_values.max(axis=1, keepdims=True))
        policy = policy / policy.sum(axis=1, keepdims=True)

        return PolicyIterationResult(
            iterations=iterations,
            state_value=state_value,
            action_value=action_value,
            converged=iterations < (self.max_iterations - 1),
            initial_value=sum([state_value[s]*p for s, p in mdp.initial_state_dist().items()]),
            policy=TabularPolicy.from_state_action_lists(mdp.state_list, mdp.action_list, policy),
        )

from msdm.core.table.tablemisc import dataclass_repr_html_MixIn
@dataclass
class PolicyIterationResult(PlanningResult,dataclass_repr_html_MixIn):
    iterations : int
    converged : bool
    state_value : dict
    initial_value : float
    action_value: dict
    policy : TabularPolicy