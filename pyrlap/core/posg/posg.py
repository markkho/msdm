from typing import Iterable, Mapping

import numpy as np

from pyrlap.core.base import State, Action, Observation, Probability
from pyrlap.core.posg.stochasticgame import StochasticGame

class PartiallyObservableStochasticGame(object):
    def __init__(self, sg : StochasticGame):
        self.sg = StochasticGame

    def observation(self, a, ns) -> Iterable[Observation]:
        raise NotImplementedError

    def observation_dist(self, a, ns) -> Mapping[Iterable[Observation], float]:
        raise NotImplementedError

    def reward(self, s=None, a=None, ns=None) -> Iterable:
        return self.sg.reward(s, a, ns)

    def reward_dist(self, s=None, a=None, ns=None) -> Mapping[Iterable, float]:
        raise self.sg.reward_dist(s, a, ns)

    def get_states(self) -> Iterable[State]:
        return self.sg.get_states()

    def get_init_state(self) -> State:
        return self.sg.get_init_state()

    def get_observations(self) -> Iterable[Observation]:
        obs = set([])
        for s, a, ns, p in self.sg.iterate_sans_prob():
            odist = self.observation_dist(a, ns)
            obs = obs.union(set(odist.keys()))
        return sorted(list(obs))

    def available_actions(self) -> Iterable[Action]:
        return self.sg.available_actions()

    def transition_dist(self, s : State, a : Action) -> Mapping[State, Probability]:
        return self.sg.transition_dist(s, a)

    def transition(self, s : State, a : Action) -> State:
        return self.sg.transition(s, a)

    def as_matrices(self) -> dict:
        mats = self.sg.as_matrices()
        oo = self.get_observations()
        aa = mats['aa']
        ss = mats['ss']
        # actions x next states x observations
        obs_func = np.zeros((len(aa), len(ss), len(oo)))
        for ai, a in enumerate(aa):
            for nsi, ns in enumerate(ss):
                odist = self.observation_dist(a, ns)
                for oi, o in enumerate(oo):
                    obs_func[ai, nsi, oi] = odist.get(o, 0.0)

        mats['of'] = obs_func
        mats['oo'] = oo
        return mats

    def is_terminal(self, s : State) -> bool:
        return self.sg.is_terminal(s)