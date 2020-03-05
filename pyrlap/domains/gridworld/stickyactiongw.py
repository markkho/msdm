from functools import lru_cache
from typing import Mapping

import numpy as np
from itertools import product

from pyrlap.domains.gridworld import GridWorld
from pyrlap.core.mdp import MDP
from pyrlap.algorithms.valueiteration import ValueIteration


class StickyActionGridWorld(MDP):
    def __init__(self, base_gw: GridWorld,
                 switch_cost=-1, init_action=None):
        self.gw = base_gw
        self.init_action = init_action
        self.switch_cost = switch_cost

    def get_init_state_dist(self):
        s0 = self.gw.get_init_state_dist()
        ss, s0 = zip(*s0.items())
        if self.init_action is None:
            aa = self.gw.available_actions()
            a0 = np.ones(len(aa)) / len(aa)
        else:
            aa = [self.init_action, ]
            a0 = [1]
        sa0 = dict(zip(product(ss, aa), np.outer(s0, a0).flatten()))
        return sa0

    def get_init_state(self):
        s0 = self.get_init_state_dist()
        ss, s0 = zip(*s0.items())
        return ss[np.random.choice(len(ss), p=s0)]

    def get_init_states(self):
        return list(self.get_init_state_dist().keys())

    def is_terminal(self, s):
        return self.gw.is_terminal(s[0])

    def is_absorbing(self, s):
        return self.gw.is_absorbing(s[0])

    def is_any_terminal(self, s):
        return self.gw.is_any_terminal(s[0])

    def is_terminal_action(self, a):
        return self.gw.is_terminal_action(a)

    def get_terminal_states(self):
        ts = self.gw.get_terminal_states()
        aa = self.gw.available_actions()
        return list(product(ts, aa))

    @lru_cache()
    def transition_reward_dist(self, s, a):
        last_a = s[1]
        trf = self.gw.transition_reward_dist(s[0], a)
        a_r = self.switch_cost if a != last_a else 0
        trf = {((ns, a), r + a_r): p for (ns, r), p in trf.items()}
        return trf

    @lru_cache()
    def reward_dist(self, s=None, a=None, ns=None):
        trf = self.transition_reward_dist(s, a)
        rdist = {}
        for (ns, r), p in trf.items():
            rdist[r] = rdist.get(r, 0.0)
            rdist[r] += p
        return rdist

    @lru_cache()
    def reward(self, s=None, a=None, ns=None):
        rd = self.reward_dist(s, a, ns)
        rs, ps = zip(*rd.items())
        return rs[np.random.choice(len(rs), p=ps)]

    def available_actions(self, s=None):
        return self.gw.available_actions(s[0] if s is not None else None)

    def get_states(self):
        return list(self.get_reachable_states())

    def plot(self, *args, **kwargs):
        return self.gw.plot(*args, **kwargs)

    def solve(self,
              discount_rate,
              softmax_temp=0.0,
              randchoose=0.0,
              **kwargs) -> ValueIteration:
        planner = ValueIteration(mdp=self,
                                 discount_rate=discount_rate,
                                 softmax_temp=softmax_temp,
                                 randchoose=randchoose,
                                 **kwargs)
        planner.solve()
        return planner

    def get_state_policy(self,
                         discount_rate,
                         softmax_temp=0.0,
                         randchoose=0.0,
                         **kwargs) -> Mapping:
        """ Returns the optimal policy marginalized over grid states """
        vi = self.solve(discount_rate, softmax_temp=softmax_temp,
                        randchoose=randchoose, **kwargs)
        pi = vi.as_matrix()
        ss = self.get_states()

        sr = vi.successor_representation(discount_rate=discount_rate,
                                         return_matrix=True)

        mats = self.as_matrices()
        s0 = mats['s0']
        aa = mats['aa']
        occ = np.einsum("sn,s->n", sr, s0)
        nocc = occ / occ.sum()

        # calc "featurized" states
        # this is based on p(s | ~s) \propto p(~s | s)rho(s)
        # where fs represents p(~s | s)
        fss = list(set([s for s, a in mats['ss']]))
        fs = []
        for sa in mats['ss']:
            fs.append([1 if sa[0] == s else 0 for s in fss])
        fs = np.array(fs)
        f_post = np.einsum("sf,s->sf", fs, nocc)
        f_post_norm = f_post.sum(axis=0)
        f_post_norm[f_post_norm == 0] = 1
        f_post = np.einsum("sf,f->sf", f_post, 1 / f_post_norm)

        flat_pi = np.einsum("sa,sf->fa", pi, f_post)
        flat_pi = {s: dict(zip(aa, ap)) for s, ap in zip(fss, flat_pi)}
        flat_pi = {s: ap for s, ap in flat_pi.items() if
                   (sum(ap.values()) != 0.0)}

        return flat_pi