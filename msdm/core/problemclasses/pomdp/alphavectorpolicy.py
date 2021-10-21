from collections import namedtuple, defaultdict
import numpy as np
from msdm.core.problemclasses.pomdp.policy import POMDPPolicy
from msdm.core.problemclasses.pomdp import TabularPOMDP
from msdm.core.problemclasses.pomdp.pomdp import Belief
from msdm.core.distributions import DictDistribution, Distribution

class AlphaVectorPolicy(POMDPPolicy):
    def __init__(
        self,
        pomdp: TabularPOMDP,
        alpha_vectors: np.array
    ):
        self.pomdp = pomdp
        self.alpha_vectors = alpha_vectors

    def initial_agentstate(self):
        return Belief(tuple(self.pomdp.state_list), tuple(self.pomdp.initial_state_vec))

    def value(self, belief):
        b = self._belief_to_vector(belief)
        return np.max(np.einsum("ds,s->d", self.alpha_vectors, b))

    def _belief_to_vector(self, belief):
        if isinstance(belief, Distribution):
            b = [belief.get(s, 0.0) for s in self.pomdp.state_list]
        elif isinstance(belief, Belief):
            ss, b = belief
            assert len(ss) == len(b)
        elif isinstance(belief, (list, tuple, np.array)):
            b = belief
        return b

    def action_values(self, belief):
        b = self._belief_to_vector(belief)
        s_dist = DictDistribution(zip(self.pomdp.state_list, b))
        action_vals = defaultdict(float)

        # one-step rewards
        for a in self.pomdp.action_list:
            for s, s_prob in s_dist.items():
                for ns, ns_prob in self.pomdp.next_state_dist(s, a).items():
                    r = self.pomdp.reward(s, a, ns)
                    action_vals[a] += r*s_prob*ns_prob

        # discounted values
        for a in self.pomdp.action_list:
            for o, o_prob in self.pomdp.predictive_observation_dist(s_dist, a).items():
                ns_dist = self.pomdp.state_estimator(s_dist, a, o)
                ns_v = self.pomdp.discount_rate * self.value(ns_dist)
                action_vals[a] += ns_v*o_prob
        return action_vals

    def action_dist(self, ag):
        av = self.action_values(ag)
        maxv = max(av.values())
        return DictDistribution.uniform([a for a, v in av.items() if v == maxv])

    def next_agentstate(self, ag, a, o):
        s_dist = DictDistribution(zip(*ag))
        ns_dist = self.pomdp.state_estimator(s_dist, a, o)
        ss = tuple(self.pomdp.state_list)
        return Belief(ss, tuple([ns_dist.get(ns, 0.) for ns in ss]))
