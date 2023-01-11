import matplotlib.pyplot as plt
from typing import Sequence
from msdm.core.utils.gridstringutils import  string_to_element_array
from msdm.core.utils.funcutils import cached_property
from frozendict import frozendict
from termcolor import colored
from collections import defaultdict

from msdm.core.mdp import TabularMarkovDecisionProcess, State

from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution, UniformDistribution

TERMINALSTATE = frozendict({'x': -1, 'y': -1})
TERMINALDIST = DeterministicDistribution(TERMINALSTATE)

class GridWorld(TabularMarkovDecisionProcess):
    """A simple gridworld domain"""
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
        self.tile_array = tile_array
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
        hash_state = lambda s: (s['x'], s['y'])
        hash_action = lambda a: (a['dx'], a['dy'])
        self._actions = sorted(actions, key=hash_action)
        self._states = sorted(states, key=hash_state)
        self._initStates = sorted(initStates, key=hash_state)
        self._absorbingStates = sorted(absorbingStates, key=hash_state)
        self._walls = sorted(walls, key=hash_state)
        self._locFeatures = locFeatures
        self.success_prob = success_prob
        if feature_rewards is None:
            feature_rewards = {'g': 0}
        if not hasattr(feature_rewards, "items"):
            feature_rewards = dict(feature_rewards)
        self._featureRewards = feature_rewards
        self.step_cost = step_cost
        self.discount_rate = discount_rate
        self._height = len(elementArray)
        self._width = len(elementArray[0])

    @property
    def state_list(self) -> Sequence[State]:
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
    def location_features(self) -> dict:
        return self._locFeatures

    @cached_property
    def feature_locations(self) -> dict:
        fl = defaultdict(list)
        for l, f in self.location_features.items():
            fl[f].append(l)
        return dict(fl)

    def is_absorbing(self, s):
        return s == TERMINALSTATE

    def next_state_dist(self, s, a) -> FiniteDistribution:
        if self.is_absorbing(s):
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
        if self.is_absorbing(s) or self.is_absorbing(ns):
            return 0.0
        f = self._locFeatures.get(ns, "")
        return self._featureRewards.get(f, 0.0) + self.step_cost

    def actions(self, s) -> Sequence:
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

    def ascii_state(self,
                    state=None,
                    ignore=('.',),
                    colors=None,
                    on_colors=None,
                    agentcolor='magenta'):
        if on_colors is None:
            on_colors = {}
        if colors is None:
            colors = {}
        tiles = []
        for _ in range(self.height):
            tiles.append([])
            for _ in range(self.width):
                tiles[-1].append(colored('  ', on_color='on_white'))

        for s in self.state_list:
            if self.is_absorbing(s):
                continue
            x, y = s['x'], s['y']
            y_ = self.height - y - 1
            icon = self.tile_array[y_][x]
            if icon in ignore:
                tile_params = {'text': f' ', 'on_color': 'on_white'}
            else:
                tile_params = {'text': f'{icon}', 'color': 'white'}
                if (icon in on_colors) or (icon in colors):
                    if icon in on_colors:
                        tile_params.update({'on_color': on_colors[icon]})
                    if icon in colors:
                        tile_params.update({'color': colors[icon]})
                elif s in self._walls:
                    tile_params.update({'color': 'white', 'on_color': 'on_grey'})
                else:
                    tile_params.update({'on_color': 'on_white'})

            if s == state:
                agent = colored('@', color=agentcolor,
                                on_color=tile_params['on_color'],
                                attrs=['bold'])
                tile = colored(**tile_params) + agent
            else:
                tile_params['text'] = tile_params['text'] + ' '
                tile = colored(**tile_params)
            tiles[y_][x] = tile

        viz = '\n'.join([''.join(row) for row in tiles])
        return viz
