from abc import abstractmethod
from typing import TypeVar, Generic, Sequence

from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.distributions import Distribution


State = TypeVar('State')
Action = TypeVar('Action')

class MarkovDecisionProcess(ProblemClass, Generic[State, Action]):
    discount_rate : float = 1.0

    @abstractmethod
    def next_state_dist(self, s: State, a: Action) -> Distribution[State]:
        pass

    @abstractmethod
    def reward(self, s: State, a: Action, ns: State) -> float:
        pass

    @abstractmethod
    def actions(self, s: State) -> Sequence[Action]:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution[State]:
        pass

    @abstractmethod
    def is_terminal(self, s: State) -> bool:
        pass
