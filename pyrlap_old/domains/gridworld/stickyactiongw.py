from functools import lru_cache
from typing import Mapping

import numpy as np
from itertools import product

from pyrlap_old.domains.gridworld import GridWorld
from pyrlap_old.core.mdp import MDP
from pyrlap_old.algorithms.valueiteration import ValueIteration


class StickyActionGridWorld(MDP):
    def __init__(self, base_gw: GridWorld,
                 switch_cost=-1, init_action=None):
        """
        Example usage with a given gridworld MDP:
        ```
        sagw = StickyActionGridWorld(gw, init_action='%')
        sagw.plot().plot_policy(policy_dict=sagw.get_state_policy(1.0))
        ```

        :param base_gw:
        :param switch_cost:
        :param init_action:
        """
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
        mats = self.as_matrices()
        aa = mats['aa']
        gs = self.calc_gridstate_weight(pi)
        gs_weight = gs['gs_weight']
        gss = gs['gss']

        flat_pi = np.einsum("sa,sf->fa", pi, gs_weight)
        flat_pi = {s: dict(zip(aa, ap)) for s, ap in zip(gss, flat_pi)}
        flat_pi = {s: ap for s, ap in flat_pi.items() if
                   (sum(ap.values()) != 0.0)}

        return flat_pi

    def get_gridstate_matrix(self, ss) -> Mapping:
        """Get matrix that relates grid states to grid + last-action states"""
        gss = self.gw.get_reachable_states()
        gs = []
        for s in ss:
            gs.append([1 if s[0] == gs else 0 for gs in gss])
        gs = np.array(gs)
        return {'gs': gs, 'gss': gss}

    def calc_gridstate_weight(self, pi_mat):
        """
        Calculates the weights on grid states under a
        gridstate + last-action state policy.
        """
        mats = self.as_matrices()
        mp = np.einsum("san,sa->sn", mats['tf'], pi_mat)
        occ = np.linalg.inv(np.eye(mp.shape[0]) - mp * mats['nt_states'])
        occ = np.einsum("sn,s->n", occ, mats['s0'])
        nocc = occ / occ.sum()

        gs_mats = self.get_gridstate_matrix(mats['ss'])
        gs = gs_mats['gs']
        g_post = np.einsum("sg,s->sg", gs, nocc)
        g_post_norm = g_post.sum(axis=0)
        g_post_norm[g_post_norm == 0] = 1
        g_post = np.einsum("sg,g->sg", g_post, 1/g_post_norm)
        return {'gs_weight': g_post, 'gss': gs_mats['gss']}

