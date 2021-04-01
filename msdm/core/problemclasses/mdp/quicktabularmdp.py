from msdm.core.distributions import Distribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, State, Action
from typing import Callable, Iterable

class QuickTabularMDP(TabularMarkovDecisionProcess):
    def __init__(
        self,
        next_state_dist: Callable[[State, Action], Distribution],
        reward: Callable[[State, Action, State], float],
        actions: Callable[[State], Iterable],
        initial_state_dist: Callable[[], Distribution],
        is_terminal: Callable[[State], bool],
        discount_rate=1.0,
        hash_state=None,
        hash_action=None
    ):
        self._next_state_dist = next_state_dist
        self._reward = reward
        self._actions = actions
        self._initial_state_dist = initial_state_dist
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
