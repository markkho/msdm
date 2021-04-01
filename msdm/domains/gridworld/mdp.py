import matplotlib.pyplot as plt
from typing import Iterable
from msdm.core.utils.gridstringutils import  string_to_element_array
from frozendict import frozendict

from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, State

from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution, UniformDistribution

TERMINALSTATE = frozendict({'x': -1, 'y': -1})
TERMINALDIST = DeterministicDistribution(TERMINALSTATE)

class GridWorld(TabularMarkovDecisionProcess):
    def __init__(self,
                 tile_array,
                 feature_rewards=None,
                 absorbing_features=("g",),
                 wall_features=("#",),
                 default_features=(".",),
                 initial_features=("s",),
                 step_cost=-1,
                 success_prob=1.0,
                 discount_rate=1.0
                 ):
        super().__init__()
        parseParams = {"colsep": "", "rowsep": "\n", "elementsep": "."}
        if not isinstance(tile_array, str):
            tile_array = "\n".join(tile_array)
        elementArray = string_to_element_array(tile_array, **parseParams)
        states = []
        walls = set()
        absorbingStates = set()
        initStates = set()
        locFeatures = {}
        for y_, row in enumerate(elementArray):
            y = len(elementArray) - y_ - 1
            for x, elements in enumerate(row):
                s = frozendict({'x': x, 'y': y})
                states.append(s)
                if len(elements) > 0:
                    f = elements[0]
                    locFeatures[s] = f
                    if f in initial_features:
                        initStates.add(s)
                    if f in absorbing_features:
                        absorbingStates.add(s)
                    if f in wall_features:
                        walls.add(s)
        states.append(TERMINALSTATE)
        actions = [
            frozendict({'dx': 0, 'dy': 0}),
            frozendict({'dx': 1, 'dy': 0}),
            frozendict({'dx': -1, 'dy': 0}),
            frozendict({'dy': 1, 'dx': 0}),
            frozendict({'dy': -1, 'dx': 0})
        ]
        self._actions = sorted(actions, key=self.hash_action)
        self._states = sorted(states, key=self.hash_state)
        self._initStates = sorted(initStates, key=self.hash_state)
        self._absorbingStates = sorted(absorbingStates, key=self.hash_state)
        self._walls = sorted(walls, key=self.hash_state)
        self._locFeatures = locFeatures
        self.success_prob = success_prob
        if feature_rewards is None:
            feature_rewards = {'g': 0}
        self._featureRewards = feature_rewards
        self.step_cost = step_cost
        self.discount_rate = discount_rate
        self._height = len(elementArray)
        self._width = len(elementArray[0])

    @property
    def state_list(self) -> Iterable[State]:
        return self._states

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def walls(self):
        return list(self._walls)

    @property
    def initial_states(self):
        return list(self._initStates)

    @property
    def absorbing_states(self):
        """
        Absorbing states are those that lead to the terminal state always
        (excluding the terminal state).
        """
        return list(self._absorbingStates)

    @property
    def location_features(self):
        return self._locFeatures

    def is_terminal(self, s):
        return s == TERMINALSTATE

    def next_state_dist(self, s, a) -> FiniteDistribution:
        if self.is_terminal(s):
            return TERMINALDIST
        if s in self.absorbing_states:
            return TERMINALDIST
        assert isinstance(s, frozendict)

        x, y = s['x'], s['y']
        ax, ay = a.get('dx', 0), a.get('dy', 0)
        nx, ny = x + ax, y + ay
        ns = frozendict({'x': nx, 'y': ny})

        if ns not in self._states:
            bdist = DeterministicDistribution(s)
        elif ns in self.walls:
            bdist = DeterministicDistribution(s)
        elif ns == s:
            bdist = DeterministicDistribution(s)
        elif self.success_prob != 1:
            bdist = DictDistribution({
                s: 1 - self.success_prob,
                ns: self.success_prob
            })
        else:
            bdist = DeterministicDistribution(ns)

        return bdist

    def reward(self, s, a, ns) -> float:
        if self.is_terminal(s) or self.is_terminal(ns):
            return 0.0
        f = self._locFeatures.get(ns, "")
        return self._featureRewards.get(f, 0.0) + self.step_cost

    def actions(self, s) -> Iterable:
        if self.is_terminal(s):
            return [frozendict({'dx': 0, 'dy': 0}), ]
        return [a for a in self._actions]

    def initial_state_dist(self) -> FiniteDistribution:
        return UniformDistribution(self.initial_states)

    def plot(self,
             all_elements=False,
             ax=None,
             figsize=None,
             figsize_multiplier=1,
             featurecolors=None,
             plot_walls=True,
             plot_initial_states=True,
             plot_absorbing_states=True
             ):
        if all_elements:
            plot_initial_states = True
            plot_absorbing_states = True
        from msdm.domains.gridworld.plotting import GridWorldPlotter
        if featurecolors is None:
            featurecolors = {
                'g': 'yellow',
                'x': 'red',
            }
        if ax is None:
            if figsize is None:
                figsize = (self.width * figsize_multiplier,
                           self.height * figsize_multiplier)
            _, ax = plt.subplots(1, 1, figsize=figsize)

        gwp = GridWorldPlotter(gw=self, ax=ax)
        gwp.plot_features(featurecolors=featurecolors)
        if plot_walls:
            gwp.plot_walls()
        if plot_initial_states:
            gwp.plot_initial_states()
        if plot_absorbing_states:
            gwp.plot_absorbing_states()
        gwp.plot_outer_box()

        return gwp

    def hash_state(self, s):
        return s['x'], s['y']

    def hash_action(self, a):
        return a['dx'], a['dy']
