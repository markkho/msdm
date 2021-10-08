from abc import abstractmethod, ABC
from typing import TypeVar

from msdm.core.problemclasses.mdp.mdp import \
    MarkovDecisionProcess, State, Action
from msdm.core.distributions import Distribution

Observation = TypeVar('Observation')

class PartiallyObservableMDP(MarkovDecisionProcess):
    """
    POMDPs as described in Kaelbling et al. (1998).
    """
    @abstractmethod
    def observation_dist(self, a : Action, ns : State) -> Distribution[Observation]:
        pass
