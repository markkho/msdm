import collections
import heapq
import random

from msdm.core.algorithmclasses import Plans, Result
from msdm.core.assignment import AssignmentSet, AssignmentMap
from msdm.core.problemclasses.mdp import DeterministicShortestPathProblem
from msdm.core.problemclasses.mdp.policy.partialpolicy import PartialPolicy

def reconstruct_path(camefrom, start, terminal_state):
    '''
    Reconstructs a path using a dictionary mapping states
    to the state that preceded them.
    '''
    path = [terminal_state]
    while path[-1] != start:
        path.append(camefrom[path[-1]])
    return path[::-1]

def path_to_policy(path):
    '''
    Converts a path (a sequence of states from a start to a goal) into a policy.
    '''
    return PartialPolicy(AssignmentMap([
        (s, AssignmentMap([(ns, 1)]))
        for s, ns in zip(path[:-1], path[1:])
    ]))

class BreadthFirstSearch(Plans):
    def plan_on(mdp: DeterministicShortestPathProblem):
        start = mdp.initial_state()

        queue = collections.deque([start])

        visited = AssignmentSet()
        camefrom = AssignmentMap()

        while queue:
            s = queue.popleft()

            if mdp.is_terminal(s):
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    policy=path_to_policy(path),
                    visited=visited,
                )

            visited.add(s)

            for a in mdp.actions(s):
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
    def plan_on(mdp: DeterministicShortestPathProblem, *, heuristic_value=lambda s: 0, seed=None):
        rnd = random.Random(seed)

        # Every queue entry is a pair of
        # - a tuple of priorities/costs (the cost-to-go, cost-so-far, and a random tie-breaker)
        # - the state
        queue = []
        start = mdp.initial_state()
        heapq.heappush(queue, ((-heuristic_value(start), 0, rnd.random()), start))

        visited = AssignmentSet()
        camefrom = AssignmentMap()

        while queue:
            (f, g, _), s = heapq.heappop(queue)

            if mdp.is_terminal(s):
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    policy=path_to_policy(path),
                    visited=visited,
                )

            visited.add(s)

            for a in mdp.actions(s):
                ns = mdp.next_state(s, a)
                if ns not in visited and ns not in [el[-1] for el in queue]:
                    ng = g - mdp.reward(s, a, ns)
                    nf = ng - heuristic_value(ns)
                    heapq.heappush(queue, ((nf, ng, rnd.random()), ns))
                    camefrom[ns] = s
