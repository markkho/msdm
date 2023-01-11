from abc import abstractmethod
from typing import TypeVar, Generic, Sequence, Set

from msdm.core.distributions import Distribution
from msdm.core.utils.funcutils import method_cache

State = TypeVar('State')
Action = TypeVar('Action')

class MarkovDecisionProcess(Generic[State, Action]):
    discount_rate : float = 1.0
    _state_list : Sequence[State]
    _action_list : Sequence[Action]

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
    def is_absorbing(self, s: State) -> bool:
        pass

    @method_cache
    def reachable_states(self, max_states=float('inf')) -> Set[State]:
        S0 = {e for e, p in self.initial_state_dist().items() if p > 0}
        frontier = set(S0)
        visited = set(S0)
        while len(frontier) > 0:
            if len(visited) >= max_states:
                break
            s = frontier.pop()
            for a in self.actions(s):
                for ns, prob in self.next_state_dist(s, a).items():
                    if prob == 0:
                        continue
                    if ns not in visited and not self.is_absorbing(ns):
                        frontier.add(ns)
                    visited.add(ns)
        return visited