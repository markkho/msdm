# encoding: utf-8
# cython: profile=False

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
import warnings

import numpy as np

from mdp_lib import GridWorld
from valueheuristic import ValueHeuristic

cdef class ForwardSearchSparseSampling(object):

    cdef public object mdp, valueheuristic, expansion_policy
    cdef float discount_rate, value_error
    cdef public int max_depth, _fsss_calls
    cdef long sample_count, iter_print
    cdef str transition_mode
    cdef public dict lower_sa, upper_sa, lower_s, upper_s, visits_s
    cdef dict ns_samples, ns_sampleprobs
    cdef set root_search_nodes
    cdef bint break_ties_randomly

    def __init__(self, mdp, discount_rate=.99, value_error=1.0,
                 transition_mode='distribution',
                 break_ties_randomly=True,
                 valueheuristic=None,
                 expansion_policy=None,
                 max_depth=None,
                 sample_count=None,
                 iter_print=500):
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
        iter_print
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
            if transition_mode == 'sampling':
                sc_a = vmax ** 2 / lambda_ ** 2
                sc_b = 2 * max_depth * math.log(
                    (n_actions * max_depth * vmax ** 2) / lambda_ ** 2, 2)
                sc_c = math.log(rmax / lambda_, 2)
                sample_count = sc_a * (sc_b + sc_c)
            else:
                sample_count = 10

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
        self.root_search_nodes = set([])
        self.break_ties_randomly = break_ties_randomly
        self.expansion_policy = expansion_policy

        self.iter_print = iter_print
        self._fsss_calls = 0

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

    def __richcmp__(self, other, op):
        if isinstance(other, self.__class__):
            if op == 2:
                return hash(self) == hash(other)
            elif op == 3:
                return hash(self) != hash(other)
        if op == 3:
            return True
        else:
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
    cdef str _get_max_upper_a(self, int d, tuple s):
        cdef float max_av, av
        cdef list max_a
        cdef str a

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
            return max_a[np.random.choice(len(max_a))]
        else:
            return max_a[0]

    cdef str _get_max_lower_a(self, int d, tuple s):
        cdef float max_av, av
        cdef list max_a
        cdef str a

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
            return max_a[np.random.choice(len(max_a))]
        else:
            return max_a[0]

    cdef tuple _get_max_valdiff_ns(self, int d, tuple s, str a):
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

    cdef str _get_max_valdiff_a(self, int d, tuple s):
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

    # cdef void _init_depth_state(self, int d, tuple s):
    def _init_depth_state(self, d, s):
        """
        For a depth/state, initializes:
         - visit count
         - upper/lower value for each action
         - upper/lower value for next states (or samples of next states)
         - samples of next states
        """
        # cdef str a
        # cdef set ns_samp
        # cdef dict ns_dist
        # cdef float low_v, upp_v, p
        # cdef tuple ns
        # cdef int ctot

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

    cdef void _retract_node(self, int d, tuple s):
        cdef str a

        del self.visits_s[(d, s)]
        for a in self.mdp.available_actions(s):
            del self.lower_sa[(d, s, a)]
            del self.upper_sa[(d, s, a)]
            del self.ns_samples[(d, s, a)]
            del self.ns_sampleprobs[(d, s, a)]

    cdef float _calc_lower_future_val(self, int d, tuple s, str a):
        """
        :param d: depth
        :param s: hashed state
        :param a: action
        :return:
        """
        cdef float lower_val, ns_prob, ns_r, ns_val
        cdef tuple ns

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

    cdef float _calc_upper_future_val(self, int d, tuple s, str a):
        """
        :param d: depth
        :param s: hashed state
        :param a: action
        :return:
        """
        cdef float upper_val, ns_prob, ns_r, ns_val
        cdef tuple ns

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

    cdef void _clear_solved_nodes(self):
        cdef int start_n, d, nd
        cdef dict parent_graph, solved_node_a
        cdef tuple s, ns
        cdef str a, max_a
        cdef bint clearable
        cdef set to_clear, next_states, parents

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
        while True:
            init_clear_count = len(to_clear)
            for (nd, ns), parents in parent_graph.iteritems():
                if (nd, ns) in self.root_search_nodes:
                    continue

                clearable = True
                #only clear nodes where all parents have abandoned it
                for (d, s) in parents:
                    if (d, s) in to_clear:
                        pass
                    elif (d, s) in solved_node_a:
                        max_a = solved_node_a[(d, s)]
                        if ns in self.ns_samples[(d, s, max_a)]:
                            clearable = False
                            break
                        if (d, s) in self.root_search_nodes:
                            clearable = False
                            break
                    else:
                        clearable = False
                        break
                if clearable:
                    to_clear.add((nd, ns))
            if init_clear_count == len(to_clear):
                break

        for d, s in to_clear:
            if (d, s) not in self.visits_s:
                continue
            self._retract_node(d, s)
        logger.debug("reduced %d nodes to %d" % (start_n, len(self.visits_s)))

    def clear_solved_nodes(self):
        self._clear_solved_nodes()


    def set_root_search_trajectory(self, traj, d=None):
        if d is None:
            d = self.max_depth
        for s, a in traj:
            self.root_search_nodes.add((d, s))
            d -= 1

    # ========================================#
    #                                         #
    #                                         #
    #   Main Forward Search Sparse Sampling   #
    #                                         #
    #                                         #
    # ========================================#
    def fsss(self, s, d, a=None,
             initial_action_rule="max_upper_a"):
        return self._fsss(s=s, d=d, a=a,
                          initial_action_rule=initial_action_rule)

    cdef void _fsss(self, tuple s, int d,
                    str a=None,
                    str initial_action_rule="max_upper_a",
                    bint continue_recursion=True):
        """
        :param s: Hashed state
        :param d: Depth
        :return: None
        """
        cdef str a_, max_a, max_lower_a, max_upper_a
        cdef tuple ns
        cdef float low_val, upp_val
        self._fsss_calls += 1

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
            # note that the upper and lower bounds will be always be equal here
            for a_ in self.mdp.available_actions(s):
                self.lower_sa[(d, s, a_)] = self._calc_lower_future_val(d, s, a_)
                self.upper_sa[(d, s, a_)] = self._calc_upper_future_val(d, s, a_)
            max_a = self._get_max_upper_a(d, s)

            self.lower_s[(d, s)] = self.lower_sa[(d, s, max_a)]
            self.upper_s[(d, s)] = self.upper_sa[(d, s, max_a)]
            return

        #=====================================================================#
        # This part of the algorithm is key for how it explores the tree.
        #
        # It needs to select an action and next state based on some criteria.
        # The selection rule described in Walsh, Goschin & Littman (2010)
        # selects the action with the max upper-bound. Then out of the set of
        # next possible states it selects the state with the widest bounds. We
        # denote that strategy "max_upper_a".
        #
        # The first time that a node is visited, we can give it a good guess
        # that effectively serves as an adaptive heuristic. We can have it use
        # the 'expansion_policy' to do thi.
        #
        if a is None and self.visits_s[(d, s)] == 0:
            if initial_action_rule == 'max_upper_a':
                max_a = self._get_max_upper_a(d, s)
                a = max_a
            elif initial_action_rule == 'expansion_policy':
                a = self.expansion_policy.get_action(s)
            # elif initial_action_rule == 'lower_a_softmax':
            #     actions = self.mdp.available_actions(s)
            #     avs = np.array([self.lower_sa[(d, s, a)] for a in actions])
            #     ps = np.exp(avs/softmax_temp)
            #     ps = ps/np.sum(ps)
            #     a = np.random.choice(actions, p=ps)
        elif a is None and self.visits_s[(d, s)] > 0:
            max_a = self._get_max_upper_a(d, s)
            a = max_a
            # #this takes the highest upper bound
            # max_a = self._get_max_upper_a(d, s)
            # a = max_a
            #
            # if use_expansion_policy and self.visits_s[(d, s)] == 0:
            #     a = self.expansion_policy.get_action(s)
            # elif action_selection_rule == 'max_upper_a':

        # This is important, so we shouldn't play around with it
        ns = self._get_max_valdiff_ns(d, s, a)

        # evaluate forward search sparse sampler at next depth
        if continue_recursion:
            self._fsss(ns, d - 1,
                       a=None,
                       initial_action_rule=initial_action_rule)

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

    def seed_trajectory(self, traj, d=None):
        if d is None:
            d = self.max_depth

        d = d - len(traj) + 1
        #need to do 2 passes
        for step in reversed(traj):
            s, a = step[0], step[1]
            self._fsss(s=s, d=d, a=a,
                       initial_action_rule="max_upper_a",
                       continue_recursion=False)
            d += 1
        # for step in traj:
        #     s, a = step[0], step[1]
        #     self._fsss(s, d, a, "max_upper_a", False)
        #     d -=1

    # ========================================#
    #                                         #
    #                                         #
    #            Search Functions             #
    #                                         #
    #                                         #
    # ========================================#
    def search(self, init_state,
               depth=None,
               termination_condition="find_best_action",
               max_iterations=1000,
               max_fsss_calls=None,
               initial_action_rule='max_upper_a',
               **kwargs):
        """

        :param init_state:
        :param termination_condition: "find_best_action", "converged_value"
        :return:
        """
        self._fsss_calls = 0
        if max_fsss_calls is None:
            max_fsss_calls = np.inf

        def run_fsss(s, d, a):
            self._fsss(s, d, a,
                       initial_action_rule=initial_action_rule)

        #don't do any additional searching from a root node
        if (depth, init_state) in self.root_search_nodes:
            logger.debug("%s previously searched" % str((depth, init_state)))
            return

        if depth is None:
            depth = self.max_depth

        if (depth, init_state) not in self.lower_s:
            self._init_depth_state(depth, init_state)

        if termination_condition == "find_best_action":
            self._search_find_best_action(init_state,
                                          depth,
                                          max_iterations,
                                          run_fsss,
                                          **kwargs
                                          )

        elif termination_condition == "converged_values":
            self._search_converged_values(init_state,
                                          depth,
                                          max_iterations,
                                          run_fsss,
                                          **kwargs)

        elif termination_condition == "decoupled_values":
            self._search_decouple_values(init_state,
                                         depth,
                                         max_iterations,
                                         max_fsss_calls,
                                         run_fsss,
                                         **kwargs)

        elif termination_condition == "fixed_iterations":
            self._search_fixed_iterations(init_state,
                                          depth,
                                          max_iterations,
                                          run_fsss,
                                          **kwargs)

        else:
            raise ValueError("Invalid termination condition")

        self.root_search_nodes.add((depth, init_state))


    cdef bint _max_action_decoupled(self, int d, tuple s):
        cdef bint solved
        cdef str max_a
        cdef float max_a_lower_bound, a_upper_bound

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

    def _get_coupled_action_bounds(self, d, s):
        bounds =  [(b, a) for a, b in self.get_bounds(d, s).iteritems()]
        bounds.sort()

        if len(bounds) == 1:
            return {}
        
        coupled = set([])

        for ai in xrange(len(bounds)):
            if ai == 0:
                if bounds[ai + 1][0][0] < bounds[ai][0][1]:
                    coupled.add(ai+1)
                    coupled.add(ai)
            elif ai == (len(bounds) - 1):
                if bounds[ai][0][0] < bounds[ai - 1][0][1]:
                    coupled.add(ai)
                    coupled.add(ai - 1)
            else:
                if bounds[ai + 1][0][0] < bounds[ai][0][1]:
                    coupled.add(ai+1)
                    coupled.add(ai)
                if bounds[ai][0][0] < bounds[ai - 1][0][1]:
                    coupled.add(ai)
                    coupled.add(ai - 1)

        return {bounds[ai][1] : bounds[ai][0] for ai in coupled}

    # cdef bint _bounds_decoupled(self, int d, tuple s,
    #                             float inter_a_bound_error,
    #                             float intra_a_bound_error):
    #     cdef list bounds
    #     cdef int decouplings, ai
    #
    #
    #     bounds = self.get_bounds(d, s).values()
    #     bounds.sort()
    #     decouplings = 0
    #     for ai in xrange(len(bounds)):
    #         if ai == 0:
    #             if bounds[ai + 1][0] > bounds[ai][1]:
    #                 decouplings += 1
    #                 continue
    #         elif ai == (len(bounds) - 1):
    #             if (bounds[ai][0] - bounds[ai - 1][1]) >= inter_a_bound_error:
    #                 decouplings += 1
    #                 continue
    #         else:
    #             if (bounds[ai + 1][0] - bounds[ai][1]) >= inter_a_bound_error \
    #                and \
    #                (bounds[ai][0] - bounds[ai - 1][1]) >= inter_a_bound_error:
    #                 decouplings += 1
    #                 continue
    #
    #         if (bounds[ai][1] - bounds[ai][0]) < intra_a_bound_error:
    #             decouplings += 1
    #             continue
    #
    #     if decouplings == len(bounds):
    #         return True
    #     else:
    #         return False

    def _search_decouple_values(self,
                                s,
                                d,
                                max_iterations,
                                max_fsss_calls,
                                run_fsss,
                                inter_a_bound_error=None,
                                intra_a_bound_error=None,
                                post_decoupling_iterations=100):

        available_actions = self.mdp.available_actions(s)
        i = 0
        decoupled = False
        while not decoupled:
            a_bd = self._get_coupled_action_bounds(d, s)

            if len(a_bd) == 0:
                self._log_bounds(d, s, i)
                decoupled = True
                break
            if (i % self.iter_print == 0) or ((i + 1) >= max_iterations):
                self._log_bounds(d, s, i)
            if ((i + 1) >= max_iterations):
                break
            if (self._fsss_calls >= max_fsss_calls):
                break
            i += 1

            a = max(a_bd.iterkeys(), key=lambda a: a_bd[a][1]-a_bd[a][0])

            run_fsss(s, d, a)

        if decoupled:
            bounds = self.get_bounds(d, s, None)
            logger.debug(str(bounds))
            a_iter = {a : b[1]-b[0] for a, b in bounds.iteritems()}
            a_s, iters = zip(*a_iter.iteritems())
            iters = np.array(iters)+.00001
            iters = iters/np.sum(iters)
            iters = [int(j) for j in post_decoupling_iterations*iters]
            a_iter = dict(zip(a_s, iters))
            logger.debug(str(a_iter))

            for a, iter in a_iter.iteritems():
                run_fsss(s, d, a)
            self._log_bounds(d, s, post_decoupling_iterations+i)
        logger.debug("")

    def _search_find_best_action(self,
                                 s,
                                 d,
                                 max_iterations,
                                 run_fsss):
        # if fsss_params is None:
        #     fsss_params = {}
        available_actions = self.mdp.available_actions(s)
        i = 0
        while True:
            run_fsss(s, d, None)

            solved = self._max_action_decoupled(d, s)
            if (i % self.iter_print == 0) or solved or i > max_iterations:
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
                                 run_fsss,
                                 value_error=None):

        available_actions = self.mdp.available_actions(s)
        max_iterations = max_iterations / len(available_actions)
        for a in available_actions:
            solved = False
            i = 0
            while not solved:
                if i >= max_iterations:
                    break
                run_fsss(s, d, a)

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
                                 run_fsss,
                                 value_error=None):

        available_actions = self.mdp.available_actions(s)
        for i in xrange(max_iterations):
            for a in available_actions:
                run_fsss(s, d, a)

            if (i % self.iter_print == 0):
                self._log_bounds(d, s, i)



    # ========================================#
    #                                         #
    #                                         #
    #   Functions for accessing results       #
    #                                         #
    #                                         #
    # ========================================#

    cpdef dict get_bounds(self, int d, tuple s, str a=None):
        cdef list available_actions, action_bounds
        cdef str a_
        cdef tuple e
        cdef tuple lower_upper

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

    def unroll(self, start, steps=10, decision_rule="max",
               epsilon=.1, softmax_temp=1, return_action_data=False):
        logger.debug("-- unrolling --")
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
                
            if decision_rule == "max":
                a = self._get_max_lower_a(d, s)
            elif decision_rule == "softmax":
                actions = self.mdp.available_actions(s)
                avs = np.array([self.lower_sa[(d, s, a)] for a in actions])
                ps = np.exp(avs/softmax_temp)
                ps = ps/np.sum(ps)
                a = np.random.choice(actions, p=ps)
            ns = None
            logger.debug("Getting next state")
            for _ in xrange(self.mdp.n_actions):
                ns = self.mdp.transition(s, a)
                logger.debug("sampled next state: %s" % str(ns))
                if (d - 1, ns) not in self.lower_s:
                    logger.debug("State %s not explored in tree. Resampling."
                        % str((d-1, ns)))
                    warnings.warn(
                        "State %s not explored in tree. Resampling."
                        % str((d-1, ns)))
                    ns = None
                else:
                    break
            if ns is None:
                logger.debug("State %s not explored in tree. Resampling."
                        % str((d-1, ns)))
                raise RuntimeError(
                    "Unable to sample an explored next state from %s"
                    % str((d, s, a)))

            r = self.mdp.reward(s=s, a=a, ns=ns)
            if return_action_data:
                actions = self.mdp.available_actions(s)
                avs_low = [self.lower_sa[(d, s, a_)] for a_ in actions]
                avs_hi = [self.upper_sa[(d, s, a_)] for a_ in actions]
                visits_s = self.visits_s[(d, s)]
                sm_probs = np.array(avs_low)
                sm_probs = np.exp(sm_probs/softmax_temp)
                sm_probs = sm_probs/np.sum(sm_probs)
                action_data = {
                    "state": s,
                    "next_state": ns,
                    "action": a,
                    "available_actions": actions,
                    "action_value_bounds": zip(avs_low, avs_hi),
                    "visits_s": visits_s,
                    "depth": d,
                    "softmax_temp": softmax_temp,
                    "softmax_probs": sm_probs
                }
                traj.append((s, a, r, action_data))
            else:
                traj.append((s, a, r))
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