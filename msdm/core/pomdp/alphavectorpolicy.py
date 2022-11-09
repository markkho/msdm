from collections import namedtuple, defaultdict
import numpy as np
from msdm.core.pomdp.policy import ValueBasedTabularPOMDPPolicy
from msdm.core.pomdp import TabularPOMDP
from msdm.core.pomdp.pomdp import Action
from msdm.core.pomdp.tabularpomdp import Belief
from msdm.core.distributions import DictDistribution, Distribution

class AlphaVectorPolicy(ValueBasedTabularPOMDPPolicy):
    def __init__(
        self,
        pomdp: TabularPOMDP,
        alpha_vectors: np.array
    ):
        super().__init__(pomdp)
        self.alpha_vectors = alpha_vectors

    def value(self, b : Belief):
        b = self._belief_to_vector(b)
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

    def action_value(self, b : Belief, a : Action):
        b = self._belief_to_vector(b)
        s_dist = DictDistribution(zip(self.pomdp.state_list, b))
        aval = 0

        # one-step reward
        for s, s_prob in s_dist.items():
            for ns, ns_prob in self.pomdp.next_state_dist(s, a).items():
                r = self.pomdp.reward(s, a, ns)
                aval += r*s_prob*ns_prob

        # discounted next state reward
        for o, o_prob in self.pomdp.predictive_observation_dist(s_dist, a).items():
            ns_dist = self.pomdp.state_estimator(s_dist, a, o)
            ns_v = self.pomdp.discount_rate * self.value(ns_dist)
            aval += ns_v*o_prob
        return aval
