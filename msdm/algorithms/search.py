import collections
from functools import lru_cache
import heapq
import random
from re import M
from typing import Dict, Union, NamedTuple, Any

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

class AStarSearchNode(NamedTuple):
    '''
    Search nodes used in AStarSearch.

    NOTE: Order of fields is important because it determines how elements are ordered in the heap.
    In particular, heuristic_cost needs to be the first field, and the tie_break should be before
    any others.
    '''
    heuristic_cost: float
    tie_break: float
    cost_from_start: float
    state: Any

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
        tie_breaking_strategy='lifo',
        assert_monotone_heuristic=True,
    ):
        self.heuristic_value = heuristic_value
        self.seed = seed
        self.randomize_action_order = randomize_action_order
        assert tie_breaking_strategy in ['random', 'lifo', 'fifo']
        self.tie_breaking_strategy = tie_breaking_strategy
        self.assert_monotone_heuristic = assert_monotone_heuristic
        if seed is not None:
            assert tie_breaking_strategy == 'random' or randomize_action_order, 'Seed was supplied, but tie-breaking and action order are deterministic.'

    def plan_on(self, dsp: MarkovDecisionProcess):
        rnd = random.Random(self.seed)
        if self.randomize_action_order:
            shuffled = make_shuffled(rnd)
        else:
            shuffled = lambda list: list

        dsp = DeterministicShortestPathProblem.from_mdp(dsp)

        tie_break = 0
        if self.tie_breaking_strategy == 'lifo':
            # The heap is a min-heap, so to ensure last-in first-out
            # the tie-breaker must decrease. Since it's always
            # decreasing, later elements of equivalent value have greater priority.
            tie_break_delta = -1
        elif self.tie_breaking_strategy == 'fifo':
            # See above comment. First-in first-out requires that our tie-breaker increases.
            tie_break_delta = +1

        queue = []
        # This holds the previous best node that was added to the queue, for each state.
        # This previous best node is also the last node for a state, since we only add when a node is an improvement.
        best_in_queue_by_state = dict()
        def push(*, heuristic_cost, cost_from_start, state):
            nonlocal tie_break
            if self.tie_breaking_strategy in ['lifo', 'fifo']:
                tie_break += tie_break_delta
            else:
                tie_break = rnd.random()
            node = AStarSearchNode(heuristic_cost=heuristic_cost, tie_break=tie_break, cost_from_start=cost_from_start, state=state)
            heapq.heappush(queue, node)
            best_in_queue_by_state[node.state] = node
            return node

        # Add the initial node.
        start = dsp.initial_state()
        push(heuristic_cost=-self.heuristic_value(start), cost_from_start=0, state=start)

        visited = set([])
        camefrom = dict()
        non_monotonic_counter = 0

        while queue:
            node = heapq.heappop(queue)
            s = node.state

            # If the state has been previously visited, then this is a worse node that should be skipped.
            if s in visited:
                assert s not in best_in_queue_by_state, 'Previously visited node should not be best node.'
                continue
            else:
                # We use `is` instead of `==` to ensure the nodes are the same object instances, not just equal.
                assert best_in_queue_by_state[s] is node, 'Newly visited state should be stored as best node.'
                # Remove the reference to this node, now that it's been removed from the queue.
                del best_in_queue_by_state[s]

            # Handle the case of a goal state.
            if dsp.is_absorbing(s):
                assert node.heuristic_cost == node.cost_from_start
                path = reconstruct_path(camefrom, start, s)
                return Result(
                    path=path,
                    path_value=node.cost_from_start,
                    policy=camefrom_to_policy(path, camefrom, dsp),
                    visited=visited,
                    non_monotonic_counter=non_monotonic_counter,
                )

            # Mark the current state as visited.
            visited.add(s)

            for a in shuffled(dsp.actions(s)):
                ns = dsp.next_state(s, a)
                # We skip previously-visited states.
                if ns in visited:
                    continue
                next_cost_from_start = node.cost_from_start - dsp.reward(s, a, ns)
                # If the state has been reached before in a lower-cost node, then we skip.
                if ns in best_in_queue_by_state and best_in_queue_by_state[ns].cost_from_start <= next_cost_from_start:
                    continue
                # At this point, we've either newly reached this state, or we have reached it in
                # a lower-cost way. So, we add it to the search queue.
                next_node = push(
                    heuristic_cost=next_cost_from_start - self.heuristic_value(ns),
                    cost_from_start=next_cost_from_start,
                    state=ns,
                )
                camefrom[ns] = (s, a)

                # Checking that the heuristic is monotonic, i.e. that our previous heuristic cost was a lower bound to the current one.
                monotone = node.heuristic_cost <= next_node.heuristic_cost
                if not monotone:
                    non_monotonic_counter += 1
                if self.assert_monotone_heuristic:
                    assert monotone, f'Heuristic is non-monotonic, with previous node {node} and next node {next_node}'
