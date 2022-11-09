from msdm.core.distributions import Distribution, DeterministicDistribution
from msdm.core.mdp.mdp import MarkovDecisionProcess, \
    State, Action
from msdm.core.mdp.tabularmdp import TabularMarkovDecisionProcess
from typing import Callable, Sequence, Union

class QuickMDP(MarkovDecisionProcess):
    def __init__(
        self,
        next_state_dist: Callable[[State, Action], Distribution[State]]=None,
        *,
        reward: Union[float, Callable[[State, Action, State], float]],
        actions: Union[Sequence[Action], Callable[[State], Sequence[Action]]],
        initial_state_dist: Union[Distribution[State], Callable[[], Distribution[State]]]=None,
        is_absorbing: Callable[[State], bool],
        # Deterministic variants.
        next_state: Callable[[State, Action], State]=None,
        initial_state: State=None,
        discount_rate=1.0
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
        self._is_absorbing = is_absorbing
        self.discount_rate = discount_rate

    def next_state_dist(self, s, a):
        return self._next_state_dist(s, a)

    def reward(self, s, a, ns):
        return self._reward(s, a, ns)

    def actions(self, s):
        return self._actions(s)

    def initial_state_dist(self):
        return self._initial_state_dist()

    def is_absorbing(self, s):
        return self._is_absorbing(s)

class QuickTabularMDP(QuickMDP,TabularMarkovDecisionProcess):
    pass
