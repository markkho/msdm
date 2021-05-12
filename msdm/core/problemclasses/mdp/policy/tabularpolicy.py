from typing import Mapping, Hashable, Sequence
import numpy as np
import math

from msdm.core.problemclasses.mdp.policy.policy import Policy
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.algorithmclasses import Result

from msdm.core.distributions import DictDistribution, \
    DeterministicDistribution, UniformDistribution, SoftmaxDistribution

class TabularPolicy(dict, Policy):
    def action_dist(self, s):
        return self[s]

    @classmethod
    def from_matrix(cls, states, actions, policy_matrix: np.array):
        assert policy_matrix.shape == (len(states), len(actions))
        policy = {}
        for si, s in enumerate(states):
            adist = {}
            assert math.isclose(policy_matrix[si, :].sum(), 1), f"Sums to {policy_matrix[si, :].sum()}"
            for ai, a in enumerate(actions):
                if policy_matrix[si, ai] > 0:
                    adist[a] = policy_matrix[si, ai]
            policy[s] = DictDistribution(adist)
        return TabularPolicy(policy)

    @classmethod
    def from_q_matrix(cls, states, actions, q: np.array, inverse_temperature=float('inf')):
        assert q.shape == (len(states), len(actions))

        if inverse_temperature == float('inf'):
            atol = np.nanmax(np.abs(np.spacing(q)))
            policy_ismax = \
                np.isclose(q, np.max(q, axis=-1, keepdims=True), rtol=0.0, atol=atol)
            # per-state count of actions with max value.
            policy_counts = policy_ismax.sum(axis=-1)

            policy = {}
            for si, s in enumerate(states):
                binary_policy = policy_ismax[si]
                if policy_counts[si] == 1:
                    policy[s] = DeterministicDistribution(actions[binary_policy.argmax()])
                else:
                    policy[s] = UniformDistribution([actions[ai] for ai in np.where(binary_policy)[0]])
        else:
            pi = np.exp(inverse_temperature*(q - np.max(q, axis=-1, keepdims=True)))
            pi = pi/pi.sum(axis=-1, keepdims=True)
            policy = {}
            for si, s in enumerate(states):
                policy[s] = SoftmaxDistribution(dict(zip(actions, pi[si])))
        return TabularPolicy(policy)

    @classmethod
    def from_deterministic_map(cls, dictionary: Mapping):
        policy = {}
        for s, p in dictionary.items():
            policy[s] = DeterministicDistribution(p)
        return TabularPolicy(policy)

    def as_matrix(self, states, actions, matrix=None):
        if matrix is None:
            matrix = np.zeros((len(states), len(actions)))
        for si, s in enumerate(states):
            adist = self.action_dist(s)
            for ai, a in enumerate(actions):
                matrix[si, ai] = adist.prob(a)
        return matrix

    def evaluate_on(self, mdp: TabularMarkovDecisionProcess) -> Result:
        mats = mdp.as_matrices()
        ss, aa, s0, tf, rf, rs, nt = \
            [mats[k] for k in ['ss', 'aa', 's0', 'tf', 'rf', 'rs', 'nt']]
        pi = self.as_matrix(ss, aa)
        mp = (rs[:, None] * (tf[:, :, :] * pi[:, :, None]).sum(1)) * nt[None, :]
        s_rf = (pi[:, :, None] * tf[:, :, :] * rf[:, :, :]).sum(axis=(1, 2))
        occ = s0@np.linalg.inv(np.eye(len(s0)) - mdp.discount_rate * mp)
        v = np.linalg.solve(np.eye(len(s0)) - mdp.discount_rate * mp, s_rf)
        q = (tf[:, :, :] * (rf[:, :, :] + v[None, None, :])).sum(axis=2)

        res = Result()
        res.mdp = mdp
        res.policy = self
        res._valuevec = v
        res.value = res.V = dict(zip(ss, v))
        occ = dict(zip(ss, occ))
        res.occupancy = res.successor_representation = occ
        res._qvaluemat = q
        qf = dict()
        for si, s in enumerate(ss):
            qf[s] = dict()
            for ai, a in enumerate(aa):
                qf[s][a] = q[si, ai]
        res.action_value = res.Q = qf
        res.initial_value = s0 @ v
        return res
