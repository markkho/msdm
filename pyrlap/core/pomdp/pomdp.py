import numpy as np

from pyrlap.core.util import sample_prob_dict
from pyrlap.core.mdp import MDP

class PartiallyObservableMarkovDecisionProcess(object):
    def __init__(self, mdp: MDP):
        self.mdp = mdp

    def observation_dist(self, a, ns) -> dict:
        raise NotImplementedError

    def observation(self, a, ns):
        return sample_prob_dict(self.observation_dist(a, ns))

    def reward(self, s=None, a=None, ns=None):
        raise NotImplementedError

    def reward_dist(self, s=None, a=None, ns=None):
        raise self.mdp.reward_dist(s, a, ns)

    def reward(self, s=None, a=None, ns=None):
        return self.mdp.reward(s, a, ns)

    def get_states(self):
        return self.mdp.get_states()

    def get_init_state(self):
        return self.mdp.get_init_state()

    def get_observations(self):
        obs = set([])
        for s, a, ns, p in self.mdp.iterate_sans_prob():
            odist = self.observation_dist(a, ns)
            obs = obs.union(set(odist.keys()))
        return sorted(list(obs))

    def available_actions(self):
        return self.mdp.available_actions()

    def transition_dist(self, s, a):
        return self.mdp.transition_dist(s, a)

    def transition(self, s, a):
        return self.mdp.transition(s, a)

    def as_matrices(self):
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

    def is_terminal(self, s):
        return self.mdp.is_terminal(s)