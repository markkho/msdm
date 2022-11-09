import collections
from functools import lru_cache
import heapq
import random
from re import M
from typing import Dict

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
        path.append(camefrom[path[-1]])
    return path[::-1]

def path_to_policy(path, mdp: MarkovDecisionProcess):
    '''
    Converts a path (a sequence of states from a start to a goal) into a policy.
    '''
    policy_dict = dict(zip(path[:-1], path[1:]))
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

    def plan_on(self, mdp: DeterministicShortestPathProblem):
        rnd = random.Random(self.seed)
        if self.randomize_action_order:
            shuffled = make_shuffled(rnd)
        else:
            shuffled = lambda list: list

        start = mdp.initial_state()

        queue = collections.deque([start])

        visited = set([])
        camefrom = dict()

        while queue:
            s = queue.popleft()

            if mdp.is_absorbing(s):
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    policy=path_to_policy(path, mdp),
                    visited=visited,
                )

            visited.add(s)

            for a in shuffled(mdp.actions(s)):
                ns = mdp.next_state(s, a)
                if ns not in visited and ns not in queue:
                    queue.append(ns)
                    camefrom[ns] = s

class AStarSearch(Plans):
    """
    A* Search is an informed best-first search algorithm. It considers states in priority order
    based on the cost to reach a state and a heuristic cost.

    Here, the heuristic cost is specified by a heuristic _value_ function, so a typical
    search heuristic for the cost should be negated.
    """
    def __init__(self, *, heuristic_value=lambda s: 0, seed=None, randomize_action_order=False):
        self.heuristic_value = heuristic_value
        self.seed = seed
        self.randomize_action_order = randomize_action_order

    def plan_on(self, mdp: DeterministicShortestPathProblem):
        rnd = random.Random(self.seed)
        if self.randomize_action_order:
            shuffled = make_shuffled(rnd)
        else:
            shuffled = lambda list: list

        # Every queue entry is a pair of
        # - a tuple of priorities/costs (the cost-to-go, cost-so-far, and a random tie-breaker)
        # - the state
        queue = []
        start = mdp.initial_state()
        heapq.heappush(queue, ((-self.heuristic_value(start), 0, rnd.random()), start))

        visited = set([])
        camefrom = dict()

        while queue:
            (f, g, r), s = heapq.heappop(queue)

            if mdp.is_absorbing(s):
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    policy=path_to_policy(path, mdp),
                    visited=visited,
                )

            visited.add(s)

            for a in shuffled(mdp.actions(s)):
                ns = mdp.next_state(s, a)
                if ns not in visited and ns not in [el[-1] for el in queue]:
                    ng = g - mdp.reward(s, a, ns)
                    nf = ng - self.heuristic_value(ns)
                    heapq.heappush(queue, ((nf, ng, rnd.random()), ns))
                    camefrom[ns] = s
