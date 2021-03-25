from typing import Mapping
import numpy as np

from msdm.core.problemclasses.mdp.policy.policy import Policy
from msdm.core.problemclasses.mdp.policy.deterministic_policy import DeterministicPolicy
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.algorithmclasses import Result

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

    def evaluate_on(self,
                    mdp: TabularMarkovDecisionProcess,
                    discount_rate = None) -> Result:
        if discount_rate is None:
            try:
                discount_rate = mdp.discount_rate
            except AttributeError:
                discount_rate = 1

        mats = mdp.as_matrices()
        ss, aa, s0, tf, rf, rs, nt = [mats[k] for k in
                                      ['ss', 'aa', 's0', 'tf', 'rf', 'rs',
                                       'nt']]
        pi = np.zeros((len(ss), len(aa)))
        for si, s in enumerate(ss):
            adist = self.action_dist(s)
            for a in adist.support:
                ai = aa.index(a)
                pi[si, ai] = adist.prob(a)

        mp = (rs[:, None] * (tf[:, :, :] * pi[:, :, None]).sum(1)) * nt[None,
                                                                     :]
        s_rf = (pi[:, :, None] * tf[:, :, :] * rf[:, :, :]).sum(axis=(1, 2))
        v = np.linalg.solve(np.eye(len(s0)) - discount_rate * mp, s_rf)
        q = (tf[:, :, :] * (rf[:, :, :] + v[None, None, :])).sum(axis=2)

        res = Result()
        res.mdp = mdp
        res.policy = self
        res._valuevec = v
        vf = AssignmentMap([(s, vi) for s, vi in zip(ss, v)])
        res.value = res.V = vf
        res._qvaluemat = q
        qf = AssignmentMap()
        for si, s in enumerate(ss):
            qf[s] = AssignmentMap()
            for ai, a in enumerate(aa):
                qf[s][a] = q[si, ai]
        res.action_value = res.Q = qf
        res.initial_value = s0 @ v
        return res

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
