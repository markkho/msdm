import collections
from functools import lru_cache
import heapq
import random
from re import M
from typing import Dict, Union

from msdm.core.algorithmclasses import Plans, Result
from msdm.core.distributions import DeterministicDistribution, DictDistribution, dictdistribution
from msdm.core.mdp.deterministic_shortest_path import DeterministicShortestPathProblem
from msdm.core.mdp.mdp import MarkovDecisionProcess
from msdm.core.mdp.policy import FunctionalPolicy

def reconstruct_path(camefrom, start, terminal_state):
    '''
    Reconstructs a path using a dictionary mapping states
    to the state that preceded them.
    '''
    path = [terminal_state]
    while path[-1] != start:
        path.append(camefrom[path[-1]][0])
    return path[::-1]

def camefrom_to_policy(path, camefrom : Dict, mdp: MarkovDecisionProcess):
    '''
    Converts a path (a sequence of states from a start to a goal) into a policy.
    '''
    policy_dict = {}
    for ns in path:
        if ns in camefrom:
            s, a = camefrom[ns]
            policy_dict[s] = a
    @FunctionalPolicy
    @lru_cache(maxsize=None)
    def policy(s):
        return DeterministicDistribution(policy_dict[s])
    return policy

def make_shuffled(rnd):
    def shuffled(iterable):
        '''
        Since random.shuffle is inplace, this function makes a copy first.
        '''
        l = list(iterable)
        rnd.shuffle(l)
        return l
    return shuffled

class BreadthFirstSearch(Plans):
    def __init__(self, *, seed=None, randomize_action_order=False):
        self.seed = seed
        self.randomize_action_order = randomize_action_order

    def plan_on(self, dsp: MarkovDecisionProcess):
        rnd = random.Random(self.seed)
        if self.randomize_action_order:
            shuffled = make_shuffled(rnd)
        else:
            shuffled = lambda list: list

        dsp = DeterministicShortestPathProblem.from_mdp(dsp)

        start = dsp.initial_state()

        queue = collections.deque([start])

        visited = set([])
        camefrom = dict()

        while queue:
            s = queue.popleft()

            if dsp.is_absorbing(s):
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    policy=camefrom_to_policy(path, camefrom, dsp),
                    visited=visited,
                )

            visited.add(s)

            for a in shuffled(dsp.actions(s)):
                ns = dsp.next_state(s, a)
                if ns not in visited and ns not in queue:
                    queue.append(ns)
                    camefrom[ns] = (s, a)

class AStarSearch(Plans):
    """
    A* Search is an informed best-first search algorithm. It considers states in priority order
    based on the cost to reach a state and a heuristic cost.

    Here, the heuristic cost is specified by a heuristic _value_ function, so a typical
    search heuristic for the cost should be negated.
    """
    def __init__(
        self, *,
        heuristic_value=lambda s: 0,
        seed=None,
        randomize_action_order=False,
        tie_breaking_strategy='lifo'
    ):
        self.heuristic_value = heuristic_value
        self.seed = seed
        self.randomize_action_order = randomize_action_order
        assert tie_breaking_strategy in ['random', 'lifo', 'fifo']
        self.tie_breaking_strategy = tie_breaking_strategy

    def plan_on(self, dsp: MarkovDecisionProcess):
        rnd = random.Random(self.seed)
        if self.randomize_action_order:
            shuffled = make_shuffled(rnd)
        else:
            shuffled = lambda list: list

        dsp = DeterministicShortestPathProblem.from_mdp(dsp)

        # Every queue entry is a pair of
        # - a tuple of priorities/costs (the cost-to-go, a tie-breaker, and cost-so-far)
        # - the state
        queue = []
        start = dsp.initial_state()
        if self.tie_breaking_strategy in ['lifo', 'fifo']:
            tie_break = 0
            if self.tie_breaking_strategy == 'lifo':
                # The heap is a min-heap, so to ensure last-in first-out
                # the tie-breaker must decrease. Since it's always
                # decreasing, later elements of equivalent value have greater priority.
                tie_break_delta = -1
            else:
                # See above comment. First-in first-out requires that our tie-breaker increases.
                tie_break_delta = +1
        else:
            tie_break = rnd.random()
        heapq.heappush(queue, ((-self.heuristic_value(start), tie_break, 0), start))

        visited = set([])
        camefrom = dict()

        while queue:
            (heuristic_cost, _, cost_from_start), s = heapq.heappop(queue)

            if dsp.is_absorbing(s):
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    policy=camefrom_to_policy(path, camefrom, dsp),
                    visited=visited,
                )

            visited.add(s)

            for a in shuffled(dsp.actions(s)):
                ns = dsp.next_state(s, a)
                if ns not in visited and ns not in [el[-1] for el in queue]:
                    next_cost_from_start = cost_from_start - dsp.reward(s, a, ns)
                    next_heuristic_cost = next_cost_from_start - self.heuristic_value(ns)
                    if self.tie_breaking_strategy in ['lifo', 'fifo']:
                        tie_break += tie_break_delta
                    else:
                        tie_break = rnd.random()
                    heapq.heappush(queue, ((next_heuristic_cost, tie_break, next_cost_from_start), ns))
                    camefrom[ns] = (s, a)
