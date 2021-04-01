from msdm.core.distributions import Distribution, DeterministicDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, State, Action
from typing import Callable, Iterable, Union

class QuickTabularMDP(TabularMarkovDecisionProcess):
    def __init__(
        self,
        next_state_dist: Callable[[State, Action], Distribution[State]]=None,
        *,
        reward: Union[float, Callable[[State, Action, State], float]],
        actions: Union[Iterable[Action], Callable[[State], Iterable[Action]]],
        initial_state_dist: Union[Distribution[State], Callable[[], Distribution[State]]]=None,
        is_terminal: Callable[[State], bool],
        # Deterministic variants.
        next_state: Callable[[State, Action], State]=None,
        initial_state: State=None,
        discount_rate=1.0,
        hash_state=None,
        hash_action=None
    ):
        assert next_state_dist is not None or next_state is not None, 'Must supply a function for next states.'
        assert initial_state_dist is not None or initial_state is not None, 'Must supply a function for initial states.'
        if next_state is None:
            self._next_state_dist = next_state_dist
        else:
            self._next_state_dist = lambda s, a: DeterministicDistribution(next_state(s, a))
        self._reward = reward if callable(reward) else lambda s, a, ns: reward
        self._actions = actions if callable(actions) else lambda s: actions
        if initial_state is not None:
            self._initial_state_dist = lambda: DeterministicDistribution(initial_state)
        elif callable(initial_state_dist):
            self._initial_state_dist = initial_state_dist
        else:
            self._initial_state_dist = lambda: initial_state_dist
        self._is_terminal = is_terminal
        self.discount_rate = discount_rate
        if hash_state is None:
            hash_state = lambda s: s
        if hash_action is None:
            hash_action = lambda s: s
        self._hash_state = hash_state
        self._hash_action = hash_action

    def next_state_dist(self, s, a):
        return self._next_state_dist(s, a)

    def reward(self, s, a, ns):
        return self._reward(s, a, ns)

    def actions(self, s):
        return self._actions(s)

    def initial_state_dist(self):
        return self._initial_state_dist()

    def is_terminal(self, s):
        return self._is_terminal(s)

    def hash_state(self, s):
        return self._hash_state(s)

    def hash_action(self, a):
        return self._hash_action(a)

# test case
