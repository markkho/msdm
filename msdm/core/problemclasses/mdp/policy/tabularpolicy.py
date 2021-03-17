from typing import Mapping
import numpy as np

from msdm.core.problemclasses.mdp.policy.policy import Policy
from msdm.core.problemclasses.mdp.policy.deterministic_policy import DeterministicPolicy
from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess

from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.distributions import DiscreteFactorTable, Distribution
class TabularPolicy(Policy):
    def __init__(self, states, actions, policy_matrix=None, policy_dict=None):
        self._states = states
        self._actions = actions
        if policy_matrix is not None:
            policy_dict = AssignmentMap()
            for si, s in enumerate(states):
                policy_dict[s] = AssignmentMap()
                for ai, a in enumerate(actions):
                    if policy_matrix[si, ai] > 0:
                        policy_dict[s][a] = policy_matrix[si, ai]
        self._policydict = policy_dict

    def evaluate_on(self, mdp: MarkovDecisionProcess) -> Mapping:
        # do policy evaluation
        raise NotImplementedError

    def action_dist(self, s) -> Distribution:
        adist = self._policydict[s]
        a, p = zip(*adist.items())
        return DiscreteFactorTable(support=a, probs=p)

    @property
    def state_list(self):
        return self._states

    @property
    def action_list(self):
        return self._actions

    @property
    def policy_dict(self) -> Mapping:
        return self._policydict

    @property
    def policy_matrix(self):
        try:
            return self._policymat
        except AttributeError:
            pass
        pi = np.zeros((len(self.state_list), len(self.action_list)))
        for si, s in enumerate(self.state_list):
            adist = self.action_dist(s)
            for ai, a in enumerate(self.action_list):
                pi[si, ai] = adist.prob(a)
        self._policymat = pi
        return self._policymat

class DeterministicTabularPolicy(TabularPolicy, DeterministicPolicy):
    pass
