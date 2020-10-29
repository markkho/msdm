import unittest

from msdm.algorithms import BreadthFirstSearch, AStarSearch
from msdm.domains import GridWorld
from msdm.core.problemclasses.mdp import DeterministicShortestPathProblem

def deterministic(dist):
    '''
    Assumes the supplied distribution is deterministic and returns the deterministic value.
    '''
    for s in dist.support:
        if dist.prob(s) == 0:
            continue
        assert dist.prob(s) == 1
        return s

gw = GridWorld(
    tile_array=[
        '...',
        '.h.',
        'sxg',
    ],
    feature_rewards={'g': 0, 'x': -100, 'h': -5},
    step_cost=-1,
    termination_prob=.0
)
# HACK: GridWorld isn't deterministic, so layering that in here. Consider making a wrapper/adaptor class to do this instead?
gw.initial_state = lambda: deterministic(gw.initial_state_dist())
gw.next_state = lambda s, a: deterministic(gw.next_state_dist(s, a))


class SearchTestCase(unittest.TestCase):
    def test_bfs(self):
        res = BreadthFirstSearch.plan_on(gw)
        assert [(s['x'], s['y']) for s in res.path] == [(0, 0), (1, 0), (2, 0), (-1, -1)]

    def test_astarsearch(self):
        def make_manhattan_distance_heuristic(mdp):
            def manhattan_distance_heuristic(s):
                if mdp.is_terminal(s):
                    return 0
                goal = mdp.absorbing_states[0]
                dist = abs(s['x'] - goal['x']) + abs(s['y'] + goal['y'])
                return -dist
            return manhattan_distance_heuristic
        res = AStarSearch.plan_on(gw, heuristic_value=make_manhattan_distance_heuristic(gw))
        assert [(s['x'], s['y']) for s in res.path] == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (-1, -1)]
