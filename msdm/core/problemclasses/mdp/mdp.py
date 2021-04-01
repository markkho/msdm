from abc import abstractmethod
from collections.abc import Sequence

from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.distributions import Distribution


class MarkovDecisionProcess(ProblemClass):
    discount_rate : float = 1.0

    @abstractmethod
    def next_state_dist(self, s, a) -> Distribution:
        pass

    @abstractmethod
    def reward(self, s, a, ns) -> float:
        pass

    @abstractmethod
    def actions(self, s) -> Sequence:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def is_terminal(self, s):
        pass

