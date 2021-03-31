from abc import abstractmethod
from collections.abc import Hashable, Mapping, Iterable

from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.distributions import Distribution
from msdm.core.utils.hashdictionary import HashDictionary, DefaultHashDictionary, defaultdict2


class MarkovDecisionProcess(ProblemClass):
    @abstractmethod
    def next_state_dist(self, s, a) -> Distribution:
        pass

    @abstractmethod
    def reward(self, s, a, ns) -> float:
        pass

    @abstractmethod
    def actions(self, s) -> Iterable:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def is_terminal(self, s):
        pass

