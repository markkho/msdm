from typing import Union, Iterable, Mapping

import numpy as np

from pyrlap.core.util import sample_prob_dict
from pyrlap.core.mdp import MDP
from pyrlap.core.base import State, Action, Observation, Probability

class PartiallyObservableMarkovDecisionProcess(object):
    def __init__(self, mdp: MDP):
        self.mdp = mdp

    def initial_belief(self):
        raise NotImplementedError

    def belief_update(self, b, o):
        raise NotImplementedError

    def observation_dist(self, a, ns) -> dict:
        raise NotImplementedError

    def observation(self, a: Action, ns: State) -> Observation:
        return sample_prob_dict(self.observation_dist(a, ns))

    def reward_dist(self,
                    s : State = None,
                    a : Action = None,
                    ns : State = None) -> Mapping[float, Probability]:
        return self.mdp.reward_dist(s, a, ns)

    def reward(self,
               s : State = None,
               a : Action = None,
               ns : State = None) -> float:
        return self.mdp.reward(s, a, ns)

    def get_states(self) -> Iterable[State]:
        return self.mdp.get_states()

    def get_init_state(self) -> State:
        return self.mdp.get_init_state()

    def get_observations(self) -> Iterable[Observation]:
        obs = set([])
        for s, a, ns, p in self.mdp.iterate_sans_prob():
            odist = self.observation_dist(a, ns)
            obs = obs.union(set(odist.keys()))
        return sorted(list(obs))

    def available_actions(self) -> Iterable[Action]:
        return self.mdp.available_actions()

    def transition_dist(self, s, a) -> Mapping[State, Probability]:
        return self.mdp.transition_dist(s, a)

    def transition(self, s, a) -> State:
        return self.mdp.transition(s, a)

    def as_matrices(self) -> dict:
        mats = self.mdp.as_matrices()
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

    def is_terminal(self, s) -> bool:
        return self.mdp.is_terminal(s)

