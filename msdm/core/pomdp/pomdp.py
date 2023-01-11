from abc import abstractmethod, ABC
from typing import TypeVar
from collections import defaultdict
import numpy as np

from msdm.core.mdp.mdp import MarkovDecisionProcess, State, Action
from msdm.core.distributions import DictDistribution, Distribution

Observation = TypeVar('Observation')

class PartiallyObservableMDP(MarkovDecisionProcess):
    """
    POMDPs as described in Kaelbling et al. (1998).
    """
    @abstractmethod
    def observation_dist(self, a : Action, ns : State) -> Distribution[Observation]:
        pass

    def state_estimator(self, b: Distribution[State], a : Action, o : Observation) -> Distribution[State]:
        """
        Returns the posterior distribution over next states
        given an action, observation, and belief over previous states.
        """
        ns_dist = defaultdict(float)
        for s, s_prob in b.items():
            if s_prob == 0.0:
                continue
            for ns, ns_prob in self.next_state_dist(s, a).items():
                o_prob = self.observation_dist(a, ns).prob(o)
                ns_dist[ns] += o_prob*s_prob*ns_prob
        tot = sum(ns_dist.values())
        if tot == 0.0:
            return DictDistribution({})
        return DictDistribution({ns: p/tot for ns, p in ns_dist.items() if p > 0.0})

    def predictive_observation_dist(self, b: Distribution[State], a : Action) -> Distribution[Observation]:
        """
        Returns the predicted observation distribution for taking
        an action given a belief distribution.
        """
        o_dist = defaultdict(float)
        for s, s_prob in b.items():
            for ns, ns_prob in self.next_state_dist(s, a).items():
                for o, o_prob in self.observation_dist(a, ns).items():
                    o_dist[o] += s_prob*ns_prob*o_prob
        assert np.isclose(sum(o_dist.values()), 1)
        return DictDistribution({o: p for o, p in o_dist.items() if p > 0.0})
