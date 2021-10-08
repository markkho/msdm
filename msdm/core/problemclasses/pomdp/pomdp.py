from abc import abstractmethod, ABC
from typing import TypeVar

from msdm.core.problemclasses.mdp.mdp import \
    MarkovDecisionProcess, State, Action
from msdm.core.distributions import Distribution

Observation = TypeVar('Observation')

class PartiallyObservableMDP(MarkovDecisionProcess):
    @abstractmethod
    def observation_dist(self, a : Action, ns : State) -> Distribution[Observation]:
        pass

    @abstractmethod
    def reward(self, s : State, a : Action, ns : State) -> float:
        pass
