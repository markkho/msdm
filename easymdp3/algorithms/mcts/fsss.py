from __future__ import division

import cPickle as pickle
import math
import time
from itertools import product
import sys
import random
import sys
import os
import logging

import numpy as np

from mdp_lib import GridWorld
from valueheuristic import ValueHeuristic

class ForwardSearchSparseSampling(object):
    #todo heapify things - this will mainly help with large A or NS
    #todo  allow for s, a, ns reward functions, right now reward functions
    #sample a next state to calculate reward.

    def __init__(self, mdp, discount_rate=.99, value_error=1.0,
                 transition_mode='distribution',
                 break_ties_randomly=True,
                 valueheuristic=None,
                 expansion_policy=None,
                 max_depth=None,
                 sample_count=None):
        """
        Parameters
        ----------
        mdp
        discount_rate
        value_error
        transition_mode
        break_ties_randomly
        valueheuristic
        expansion_policy
        max_depth
        sample_count
        """
        rmax = mdp.rmax
        n_actions = mdp.n_actions
        lambda_ = value_error * (1 - discount_rate) ** 2 / 4
        delta_ = lambda_ / rmax
        if valueheuristic is None:
            valueheuristic = ValueHeuristic(rmax, -rmax, discount_rate)
        vmax = valueheuristic.vmax()

        if max_depth is None:
            max_depth = math.ceil(math.log(lambda_ / vmax, discount_rate))

        if sample_count is None:
            sc_a = vmax ** 2 / lambda_ ** 2
            sc_b = 2 * max_depth * math.log(
                (n_actions * max_depth * vmax ** 2) / lambda_ ** 2, 2)
            sc_c = math.log(rmax / lambda_, 2)
            sample_count = sc_a * (sc_b + sc_c)

        self.mdp = mdp
        self.valueheuristic = valueheuristic
        self.discount_rate = discount_rate
        self.value_error = value_error
        self.max_depth = max_depth
        self.sample_count = sample_count
        self.transition_mode = transition_mode

        self.lower_sa = {}
        self.upper_sa = {}
        self.lower_s = {}
        self.upper_s = {}
        self.visits_s = {}
        self.ns_samples = {}
        self.ns_sampleprobs = {}
        self.nodelete_nodes = set([])
        self.break_ties_randomly = break_ties_randomly
        self.expansion_policy = expansion_policy

    def size(self):
        return len(self.visits_s.values())

    def __hash__(self):
        return hash((
            self.mdp,
            self.discount_rate,
            # self.value_error,
            # self.transition_mode,
            # self.break_ties_randomly,
            self.valueheuristic,
            # self.max_depth,
            # self.sample_count
        ))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return False

    def save_data(self, superdir):
        objdir = os.path.join(superdir, str(hash(self)))
        if not os.path.exists(objdir):
            os.makedirs(objdir)
        attrs = ['lower_sa',
                 'upper_sa',
                 'lower_s',
                 'upper_s',
                 'visits_s',
                 'ns_samples',
                 'ns_sampleprobs']
        logger.debug("Saving FSSS #%d" % hash(self))
        for attr in attrs:
            attrpath = os.path.join(objdir, attr+'.pkl')
            with open(attrpath, 'wb') as f:
                pickle.dump(getattr(self, attr), f, protocol=2)
                logger.debug("\t%s pickled" % attr)

    def load_data(self, superdir):
        objdir = os.path.join(superdir, str(hash(self)))
        attrs = ['lower_sa',
                 'upper_sa',
                 'lower_s',
                 'upper_s',
                 'visits_s',
                 'ns_samples',
                 'ns_sampleprobs']
        logger.debug("Loading FSSS #%d" % hash(self))
        for attr in attrs:
            attrpath = os.path.join(objdir, attr+'.pkl')
            with open(attrpath, 'rb') as f:
                val = pickle.load(f)
                setattr(self, attr, val)
                logger.debug("\t%s unpickled" % attr)

    #=========================================#
    #                                         #
    #                                         #
    #            Helper Functions             #
    #                                         #
    #                                         #
    #=========================================#
    def _get_max_upper_a(self, d, s):
        max_av = -np.inf
        max_a = []
        for a in self.mdp.available_actions(s):
            av = self.upper_sa[(d, s, a)]
            if av > max_av:
                max_av = av
                max_a = [a, ]
            elif av == max_av:
                max_a.append(a)
        
        if self.break_ties_randomly:
            max_a = np.random.choice(max_a)
            return max_a
        else:
            return max_a[0]

    def _get_max_lower_a(self, d, s):
        max_av = -np.inf
        max_a = []
        for a in self.mdp.available_actions(s):
            av = self.lower_sa[(d, s, a)]
            if av > max_av:
                max_av = av
                max_a = [a, ]
            elif av == max_av:
                max_a.append(a)
        
        if self.break_ties_randomly:
            max_a = np.random.choice(max_a)
            return max_a
        else:
            return max_a[0]

    def _get_max_valdiff_ns(self, d, s, a):
        max_ns_vd = -np.inf
        max_ns = []
        for ns in self.ns_samples[(d, s, a)]:
            ns_vd = self.upper_s[(d - 1, ns)] - self.lower_s[(d - 1, ns)]
            if ns_vd > max_ns_vd:
                max_ns_vd = ns_vd
                max_ns = [ns, ]
            elif ns_vd == max_ns_vd:
                max_ns.append(ns)

        if self.break_ties_randomly:
            max_ns = max_ns[np.random.choice(len(max_ns))]
            return max_ns
        else:
            return max_ns[0]

    def _get_max_valdiff_a(self, d, s):
        if (d, s) not in self.visits_s:
            self._init_depth_state(d, s)
            return self.mdp.available_actions(s)[0]

        max_av_diff = -np.inf
        max_a = []
        for a in self.mdp.available_actions(s):
            av_low = self.lower_sa[(d, s, a)]
            av_upp = self.upper_sa[(d, s, a)]
            av_diff = av_upp - av_low
            if (av_upp - av_low) > max_av_diff:
                max_av_diff = (av_upp - av_low)
                max_a = [a, ]
            elif (av_upp - av_low) == max_av_diff:
                max_a.append(a)
        if self.break_ties_randomly:
            max_a = max_a[np.random.choice(len(max_a))]
            return max_a
        else:
            return max_a[0]

    def _init_depth_state(self, d, s):
        """
        For a depth/state, initializes:
         - visit count
         - upper/lower value for each action
         - upper/lower value for next states (or samples of next states)
         - samples of next states
        """
        self.visits_s[(d, s)] = 0
        for a in self.mdp.available_actions(s):
            self.lower_sa[(d, s, a)] = self.valueheuristic.vmin(s)
            self.upper_sa[(d, s, a)] = self.valueheuristic.vmax(s)

            ns_samp = set([])
            if self.transition_mode == 'distribution':
                ns_dist = self.mdp.transition_dist(s, a)
                for ns, p in ns_dist.iteritems():
                    low_v = self.lower_s.get((d - 1, ns),
                                             self.valueheuristic.vmin(ns))
                    upp_v = self.upper_s.get((d - 1, ns),
                                             self.valueheuristic.vmax(ns))
                    self.lower_s[(d - 1, ns)] = low_v
                    self.upper_s[(d - 1, ns)] = upp_v
                    ns_samp.add(ns)

            elif self.transition_mode == 'sampling':
                ns_dist = {}
                for c in xrange(self.sample_count):
                    ns = self.mdp.transition(s, a)
                    self.lower_s[(d - 1, ns)] = self.valueheuristic.vmin(ns)
                    self.upper_s[(d - 1, ns)] = self.valueheuristic.vmax(ns)
                    ns_samp.add(ns)
                    p = ns_dist.get(ns, 0)
                    ns_dist[ns] = p+1
                ctot = self.sample_count
                ns_dist = {ns: p/ctot for ns, p in ns_dist.iteritems()}

            self.ns_samples[(d, s, a)] = ns_samp
            self.ns_sampleprobs[(d, s, a)] = ns_dist

    def _retract_node(self, d, s):
        del self.visits_s[(d, s)]
        for a in self.mdp.available_actions(s):
            del self.lower_sa[(d, s, a)]
            del self.upper_sa[(d, s, a)]
            del self.ns_samples[(d, s, a)]
            del self.ns_sampleprobs[(d, s, a)]

    def _calc_lower_future_val(self, d, s, a):
        """
        :param d: depth
        :param s: hashed state
        :param a: action
        :return:
        """
        lower_val = 0
        for ns in self.ns_samples[(d, s, a)]:
            ns_prob = self.ns_sampleprobs[(d, s, a)][ns]
            ns_r = self.mdp.reward(s=s, a=a, ns=ns)
            if d > 1:
                ns_val = self.lower_s[(d - 1, ns)]
                lower_val += (ns_r + self.discount_rate*ns_val) * ns_prob
            else:
                lower_val += ns_r * ns_prob
        return lower_val

    def _calc_upper_future_val(self, d, s, a):
        """
        :param d: depth
        :param s: hashed state
        :param a: action
        :return:
        """
        upper_val = 0
        for ns in self.ns_samples[(d, s, a)]:
            ns_prob = self.ns_sampleprobs[(d, s, a)][ns]
            ns_r = self.mdp.reward(s=s, a=a, ns=ns)
            if d > 1:
                ns_val = self.upper_s[(d - 1, ns)]
                upper_val += (ns_r + self.discount_rate*ns_val) * ns_prob
            else:
                upper_val += ns_r * ns_prob
        return upper_val

    def clear_solved_nodes(self):
        start_n = len(self.visits_s)
        parent_graph = {}
        for (d, s, a), next_states in self.ns_samples.iteritems():
            if d == 1:
                continue
            for ns in next_states:
                parents = parent_graph.get((d - 1, ns), set([]))
                parents.add((d, s))
                parent_graph[(d - 1, ns)] = parents

        solved_node_a = {}
        for (d, s) in self.visits_s:
            if self._max_action_decoupled(d, s):
                solved_node_a[(d, s)] = self._get_max_upper_a(d, s)

        to_clear = set([])
        for (nd, ns), parents in parent_graph.iteritems():
            if (nd, ns) in self.nodelete_nodes:
                continue

            clearable = True
            #only clear nodes where all parents have abandoned it
            for (d, s) in parents:
                if (d, s) in solved_node_a:
                    max_a = solved_node_a[(d, s)]
                    if ns in self.ns_samples[(d, s, max_a)]:
                        clearable = False
                        break
                    if (d, s) in self.nodelete_nodes:
                        clearable = False
                        break
                else:
                    clearable = False
                    break
            if clearable:
                to_clear.add((nd, ns))

        for d, s in to_clear:
            if (d, s) not in self.visits_s:
                continue
            self._retract_node(d, s)
        logger.debug("reduced %d nodes to %d" % (start_n, len(self.visits_s)))

    def set_nonretractable_nodes(self, traj, d=None):
        if d is None:
            d = self.max_depth
        for s, a in traj:
            self.nodelete_nodes.add((d, s))
            d -= 1

    # ========================================#
    #                                         #
    #                                         #
    #   Main Forward Search Sparse Sampling   #
    #                                         #
    #                                         #
    # ========================================#

    def fsss(self, s, d, a=None, use_expansion_policy=False):
        """
        :param s: Hashed state
        :param d: Depth
        :return: None
        """

        if (d, s) not in self.visits_s:
            self._init_depth_state(d, s)

        if self.mdp.is_terminal(s):
            for a_ in self.mdp.available_actions(s):
                self.lower_sa[(d, s, a_)] = self.mdp.terminal_state_reward
                self.upper_sa[(d, s, a_)] = self.mdp.terminal_state_reward
            self.lower_s[(d, s)] = self.mdp.terminal_state_reward
            self.upper_s[(d, s)] = self.mdp.terminal_state_reward
            return

        if d == 1:
            for a_ in self.mdp.available_actions(s):
                self.lower_sa[(d, s, a_)] = self._calc_lower_future_val(d, s, a_)
                self.upper_sa[(d, s, a_)] = self._calc_upper_future_val(d, s, a_)
            max_a = self._get_max_upper_a(d, s)
            self.lower_s[(d, s)] = self.lower_sa[(d, s, max_a)]
            self.upper_s[(d, s)] = self.upper_sa[(d, s, max_a)]
            return

        # get action, and most unbounded next state
        if a is None and use_expansion_policy and self.visits_s[(d, s)] == 0:
            a = self.expansion_policy.get_action(s)
        elif a is None:
            max_a = self._get_max_upper_a(d, s)
            a = max_a

        ns = self._get_max_valdiff_ns(d, s, a)

        # evaluate forward search sparse sampler at next depth
        self.fsss(ns, d - 1,
                  use_expansion_policy=use_expansion_policy)

        # update counts and lower/upper state-action values
        self.visits_s[(d, s)] += 1
        low_val = self._calc_lower_future_val(d, s, a)
        upp_val = self._calc_upper_future_val(d, s, a)
        self.lower_sa[(d, s, a)] = low_val
        self.upper_sa[(d, s, a)] = upp_val

        #set upper/lower state value for this depth/state
        max_lower_a = self._get_max_lower_a(d, s)
        max_upper_a = self._get_max_upper_a(d, s)
        self.lower_s[(d, s)] = self.lower_sa[(d, s, max_lower_a)]
        self.upper_s[(d, s)] = self.upper_sa[(d, s, max_upper_a)]

    # ========================================#
    #                                         #
    #                                         #
    #            Search Functions             #
    #                                         #
    #                                         #
    # ========================================#
    def search(self, init_state,
               termination_condition="find_best_action",
               depth=None,
               max_iterations=1000,
               use_expansion_policy=False,
               **kwargs):
        """

        :param init_state:
        :param termination_condition: "find_best_action", "converged_value"
        :return:
        """
        self.nodelete_nodes.add((depth, init_state))

        if depth is None:
            depth = self.max_depth

        if (depth, init_state) not in self.lower_s:
            self._init_depth_state(depth, init_state)

        if termination_condition == "find_best_action":
            self._search_find_best_action(init_state,
                                          depth,
                                          max_iterations,
                                          use_expansion_policy,
                                          **kwargs
                                          )

        elif termination_condition == "converged_values":
            self._search_converged_values(init_state,
                                          depth,
                                          max_iterations,
                                          use_expansion_policy,
                                          **kwargs)

        elif termination_condition == "decoupled_values":
            self._search_decouple_values(init_state,
                                         depth,
                                         max_iterations,
                                         use_expansion_policy,
                                         **kwargs)

        elif termination_condition == "fixed_iterations":
            self._search_fixed_iterations(init_state,
                                   depth,
                                   max_iterations,
                                   use_expansion_policy,
                                   **kwargs)

        else:
            raise ValueError("Invalid termination condition")


    def _max_action_decoupled(self, d, s):
        solved = True
        max_a = self._get_max_upper_a(d, s)
        max_a_lower_bound = self.lower_sa[(d, s, max_a)]
        for a in self.mdp.available_actions(s):
            if max_a == a:
                continue
            a_upper_bound = self.upper_sa[(d, s, a)]
            if max_a_lower_bound < a_upper_bound:
                solved = False
                break
        return solved

    def _bounds_decoupled(self, d, s,
                          inter_a_bound_error,
                          intra_a_bound_error):
        bounds = self.get_bounds(d, s).values()
        bounds.sort()
        solved_actions = 0
        for ai in xrange(len(bounds)):
            if ai == 0:
                if (bounds[ai + 1][0] - bounds[ai][1]) >= inter_a_bound_error:
                    solved_actions += 1
                    continue
            elif ai == (len(bounds) - 1):
                if (bounds[ai][0] - bounds[ai - 1][1]) >= inter_a_bound_error:
                    solved_actions += 1
                    continue
            else:
                if (bounds[ai + 1][0] - bounds[ai][1]) >= inter_a_bound_error \
                   and \
                   (bounds[ai][0] - bounds[ai - 1][1]) >= inter_a_bound_error:
                    solved_actions += 1
                    continue

            if (bounds[ai][1] - bounds[ai][0]) < intra_a_bound_error:
                solved_actions += 1
                continue

        if solved_actions == len(bounds):
            return True
        else:
            return

    def _search_decouple_values(self,
                                s,
                                d,
                                max_iterations,
                                use_expansion_policy,
                                inter_a_bound_error=None,
                                intra_a_bound_error=None,
                                progress_file=sys.stdout
                                ):

        available_actions = self.mdp.available_actions(s)
        i = 0
        solved = False
        while not solved:
            a = self._get_max_valdiff_a(d, s)
            self.fsss(s, d, a, use_expansion_policy)
            solved = self._bounds_decoupled(d, s,
                                            inter_a_bound_error,
                                            intra_a_bound_error)
            if (i % 1000 == 0) or solved or ((i + 1) >= max_iterations):
                self._log_bounds(d, s, i)
            i += 1
            if i >= max_iterations:
                break
        logger.debug("")

    def _search_find_best_action(self,
                                 s,
                                 d,
                                 max_iterations,
                                 use_expansion_policy):
        available_actions = self.mdp.available_actions(s)
        i = 0
        while True:
            self.fsss(s, d, None, use_expansion_policy)

            solved = self._max_action_decoupled(d, s)
            if (i % 10 == 0) or solved or i > max_iterations:
                self._log_bounds(d, s, i)

            if solved:
                break

            i += 1
            if i > max_iterations:
                break

    def _search_converged_values(self,
                                 s,
                                 d,
                                 max_iterations,
                                 use_expansion_policy,
                                 value_error=None):
        available_actions = self.mdp.available_actions(s)
        max_iterations = max_iterations / len(available_actions)
        for a in available_actions:
            solved = False
            i = 0
            while not solved:
                if i >= max_iterations:
                    break
                self.fsss(s, d, a, use_expansion_policy)

                low_val = self.lower_sa[(d, s, a)]
                upp_val = self.upper_sa[(d, s, a)]
                if (upp_val - low_val) < value_error:
                    solved = True

                if (i % 100 == 0) or \
                        solved or \
                        ((i + 1) >= max_iterations):
                    self._log_bounds(d, s, i)
                i += 1

    def _search_fixed_iterations(self,
                                 s,
                                 d,
                                 max_iterations,
                                 use_expansion_policy,
                                 value_error=None):
        available_actions = self.mdp.available_actions(s)
        for i in xrange(max_iterations):
            for a in available_actions:
                self.fsss(s, d, a, use_expansion_policy)

            if (i % 1000 == 0):
                self._log_bounds(d, s, i)



    # ========================================#
    #                                         #
    #                                         #
    #   Functions for accessing results       #
    #                                         #
    #                                         #
    # ========================================#

    def get_bounds(self, d, s, a=None):
        available_actions = self.mdp.available_actions(s)
        action_bounds = []
        for a_ in available_actions:
            if (d, s, a_) not in self.lower_sa:
                self._init_depth_state(d, s)
            e = (d, s, a_)
            lower_upper = (self.lower_sa[e], self.upper_sa[e])
            action_bounds.append(lower_upper)
        return dict(zip(available_actions, action_bounds))

    def _log_bounds(self, d, s, i=None):
        max_a = self._get_max_upper_a(d, s)
        max_a_ub = self.upper_sa[(d, s, max_a)]
        res = []
        if i is not None:
            res.append("%d: " % i)

        for a_, (l, u) in self.get_bounds(d, s).iteritems():
            if u == max_a_ub:
                res.append("%s: (%.2f, %.2f)* " % (a_, l, u))
            else:
                res.append("%s: (%.2f, %.2f)  " % (a_, l, u))
        logger.debug("".join(res))

    def unroll(self, start, steps=10):
        s = start
        for d in xrange(int(self.max_depth), 0, -1):
            if (float(d), s) in self.lower_s:
                d = float(d)
                break
            if d == 1:
                raise ValueError("Start state not found")

        traj = []
        for i in xrange(steps):
            if self.mdp.is_terminal(s):
                break
            max_a = self._get_max_lower_a(d, s)
            ns = None
            for _ in xrange(self.mdp.n_actions):
                ns = self.mdp.transition(s, max_a)
                if (d - 1, ns) not in self.lower_s:
                    warn(
                        "State %s not explored in tree. Resampling."
                        % str((d-1, ns)))
                    ns = None
                else:
                    break
            if ns is None:
                raise RuntimeError(
                    "Unable to sample an explored next state from %s"
                    % str((d, s, a)))

            r = self.mdp.reward(s=s, a=max_a, ns=ns)
            traj.append((s, max_a, r))
            d -= 1
            s = ns
        return traj

#========================================#
#                                        #
#                                        #
#            Test Code                   #
#                                        #
#                                        #
#========================================#

def setup_gw(tile_features, feature_rewards, absorbing_states, init_state):
    w = len(tile_features[0])
    h = len(tile_features)
    state_features = {}
    for x, y in product(range(w), range(h)):
        state_features[(x, y)] = tile_features[h - 1 - y][x]

    params = {
        'width': w,
        'height': h,
        'state_features': state_features,
        'feature_rewards': feature_rewards,
        'absorbing_states': absorbing_states,
        'init_state': init_state
    }

    gw = GridWorld(**params)
    return gw

if __name__ == "__main__":
    start = time.time()
    tiles = [['w', 'w', 'w', 'x', 'w', 'w', 'w'],
             ['w', 'x', 'w', 'w', 'w', 'x', 'y']]
    absorbing_states = [(6, 0), ]
    feature_rewards = {
        'w': 0,
        'y': 1,
        'x': -1
    }
    init_state = (0, 0)
    gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

    fsss = ForwardSearchSparseSampling(gw,
                                       discount_rate=.95,
                                       value_error=.01,
                                       break_ties_randomly=True,
                                       max_depth=20)
    fsss.search(init_state=gw.get_init_state(),
             termination_condition="converged_values")

    traj = fsss.unroll(start=gw.get_init_state(),
                       steps=100)

    print(traj)
    print('%.2f' % (time.time() - start))

logger = logging.getLogger(__name__)