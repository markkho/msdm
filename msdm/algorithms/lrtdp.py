from functools import lru_cache
import random
import copy
import warnings
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Callable
from msdm.core.mdp import TabularMarkovDecisionProcess, FunctionalPolicy, \
        MarkovDecisionProcess, HashableState
from msdm.core.utils.dictutils import defaultdict2
from msdm.core.distributions import DictDistribution
from msdm.core.algorithmclasses import Plans, PlanningResult

class LRTDP(Plans):
    def __init__(self,
                 heuristic : Callable[[HashableState], float],
                 bellman_error_margin :float=1e-2,
                 iterations : int=int(2**30),
                 randomize_action_order : bool=False,
                 max_trial_length : int=None,
                 event_listener_class : "LRTDPEventListener"=None,
                 seed=None
                 ):
        """
        Labeled Real-Time Dynamic Programming (Bonet & Geffner 2003).

        Parameters
        ----------
        heuristic : Callable[[HashableState], float]
            State-heuristic function. If this over-estimates
            the value at all states, then the
            algorithm will converge to an optimal solution.

        bellman_error_margin : float

        iterations : int
            Number of trials of LRTDP to run.

        randomize_action_order : bool
            False by default. If set to True, then actions at
            a state are randomly ordered when that state is first
            encountered and fixed to that order subsequently. This
            ensures ties are broken randomly but consistently.

        max_trial_length : int
            By default this is infinity. The convergence properties
            of LRTDP when this is < infinity are not guaranteed.

        event_listener_class : LRTDPEventListener

        seed : int
            Random seed

        References
        ----------
        Bonet, Blai, and Hector Geffner. "Labeled RTDP:
        Improving the Convergence of Real-Time Dynamic
        Programming." ICAPS. Vol. 3. 2003.

        Notes
        -----
        This implementation mixes names from
        Bonet & Geffner 2003 and Ghallab, Nau, Traverso.
        """

        self.heuristic = heuristic
        self.bellman_error_margin = bellman_error_margin
        self.iterations = iterations
        self.randomize_action_order = randomize_action_order
        self.seed = seed
        if max_trial_length is None:
            max_trial_length = float('inf')
        self.max_trial_length = max_trial_length
        self.event_listener_class = event_listener_class

    def plan_on(self, mdp: MarkovDecisionProcess):
        self._set_up_plan_on()
        self.lrtdp(
            mdp, heuristic=self.heuristic, iterations=self.iterations
        )
        res = self._tear_down_plan_on(mdp, self.heuristic)
        return res

    def _set_up_plan_on(self):
        self.res = PlanningResult()

        if self.seed is None:
            self.res.seed = random.randint(0, int(2**30))
        else:
            self.res.seed = self.seed
        self.rng = random.Random(self.seed)

        if self.event_listener_class is not None:
            self.res.event_listener = self.event_listener_class()
        else:
            self.res.event_listener = None

    def _tear_down_plan_on(
        self,
        mdp : TabularMarkovDecisionProcess,
        heuristic
    ):
        res = self.res

        q_values = defaultdict(lambda : dict())
        policy_dict = {}
        for s in self.res.V.keys():
            policy_dict[s] = self.policy(mdp, s)
            for a in mdp.actions(s):
                q_values[s][a] = self.Q(mdp, s, a)
        res.Q = q_values

        @FunctionalPolicy
        @lru_cache(maxsize=None)
        def policy(s):
            try:
                action = policy_dict[s]
                return DictDistribution.deterministic(action)
            except KeyError:
                pass
            max_actions = []
            max_val = float('-inf')
            for a in mdp.actions(s):
                ns_dist = mdp.next_state_dist(s, a)
                val = ns_dist.expectation(
                    lambda ns : mdp.reward(s, a, ns) + mdp.discount_rate*heuristic(ns)
                ) 
                if val > max_val:
                    max_actions = [a]
                elif val == max_val:
                    max_actions.append(a)
                max_val = max(val, max_val)
            return DictDistribution.uniform(max_actions)

        res.policy = policy
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])

        #clear result
        self.res = None
        return res

    def lrtdp(self, mdp : MarkovDecisionProcess, heuristic=None, iterations=None):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        self.res.V = defaultdict2(heuristic)
        self.res.action_orders = dict()

        # Keeping track of "labels": which states have been solved
        self.res.solved = defaultdict2(lambda s: False)

        for i in range(iterations):
            if all(self.res.solved[s] for s in mdp.initial_state_dist().support):
                return
            self.lrtdp_trial(mdp, mdp.initial_state_dist().sample(rng=self.rng))
        if i == (iterations - 1):
            warnings.warn(f"LRTDP not converged after {iterations} iterations")
            self.res.converged = False
        else:
            self.res.converged = True

    def lrtdp_trial(self, mdp : MarkovDecisionProcess, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = [s, ]
        while not self.res.solved[s]:
            self._bellman_update(mdp, s)
            s = mdp.next_state_dist(s, self.policy(mdp, s)).sample(rng=self.rng)
            visited.append(s)

            # Terminal states are solved.
            if mdp.is_absorbing(s):
                self.res.solved[s] = True
            if len(visited) > self.max_trial_length:
                break
            if self.res.event_listener is not None:
                self.res.event_listener.end_of_lrtdp_timestep(locals())
        if self.res.event_listener is not None:
            self.res.event_listener.end_of_lrtdp_trial(locals())
        s = visited.pop()
        while self._check_solved(mdp, s) and visited:
            s = visited.pop()

    def _check_solved(self, mdp, s):
        # GNT Algorithm 6.18
        flag = True
        open = []
        closed = []
        if not self.res.solved[s]:
            open.append(s)
        while open:
            s = open.pop()
            closed.append(s)
            residual = self.res.V[s] - self.Q(mdp, s, self.policy(mdp, s))
            if abs(residual) > self.bellman_error_margin:
                flag = False
            else:
                for ns in mdp.next_state_dist(s, self.policy(mdp, s)).support:
                    if not self.res.solved[ns] and ns not in open and ns not in closed:
                        open.append(ns)
        if flag:
            for ns in closed:
                self.res.solved[ns] = True
        else:
            while closed:
                s = closed.pop()
                self._bellman_update(mdp, s)
        return flag

    def _bellman_update(self, mdp, s):
        '''
        Following Bonet & Geffner 2003, we only explicitly store value
        and compute Q values and the policy from it, important in computing
        the residual in _check_solved().
        '''
        self.res.V[s] = max(self.Q(mdp, s, a) for a in mdp.actions(s))

    def Q(self, mdp: MarkovDecisionProcess, s, a):
        if mdp.is_absorbing(s):
            return 0
        q = 0
        for ns, prob in mdp.next_state_dist(s, a).items():
            future = 0
            if not mdp.is_absorbing(ns):
                future = self.res.V[ns]
            q += prob * (mdp.reward(s, a, ns) + mdp.discount_rate*future)
        return q

    def policy(self, mdp, s):
        if s in self.res.action_orders:
            action_list = self.res.action_orders[s]
        else:
            if self.randomize_action_order:
                action_list = list(mdp.actions(s))
                self.rng.shuffle(action_list)
            else:
                action_list = mdp.actions(s)
            self.res.action_orders[s] = action_list
        return max(action_list, key=lambda a: self.Q(mdp, s, a))

class LRTDPEventListener(ABC):
    @abstractmethod
    def end_of_lrtdp_trial(self, localvars):
        pass
    @abstractmethod
    def end_of_lrtdp_timestep(self, localvars):
        pass
