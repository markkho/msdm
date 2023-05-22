from msdm.algorithms import BreadthFirstSearch, AStarSearch
from msdm.core.mdp import MarkovDecisionProcess
from msdm.domains import GridWorld
from msdm.tests.domains import DeterministicCounter

gw = GridWorld(
    tile_array=[
        '...',
        '.h.',
        'sxg',
        '.h.',
        '...',
    ],
    feature_rewards={'g': 0, 'x': -100, 'h': -5},
    step_cost=-1,
    discount_rate=1.0
)

def test_bfs():
    res = BreadthFirstSearch().plan_on(gw)
    assert [(s['x'], s['y']) for s in res.path] == [(0, 2), (1, 2), (2, 2), (-1, -1)]

def test_deterministic_shortest_path():
    res = BreadthFirstSearch().plan_on(DeterministicCounter(3))
    assert res.path == [0, 1, 2, 3]

def test_astarsearch():
    soln1 = [(0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (-1, -1)]
    soln2 = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (-1, -1)]
    def make_manhattan_distance_heuristic(mdp : MarkovDecisionProcess):
        def manhattan_distance_heuristic(s):
            if mdp.is_absorbing(s):
                return 0
            goal = mdp.absorbing_states[0]
            dist = abs(s['x'] - goal['x']) + abs(s['y'] - goal['y'])
            return -dist
        return manhattan_distance_heuristic
    planner = AStarSearch(
        heuristic_value=make_manhattan_distance_heuristic(gw),
        randomize_action_order=False,
        seed=42
    )
    res = planner.plan_on(gw)
    assert [(s['x'], s['y']) for s in res.path] == soln1
    planner = AStarSearch(
        heuristic_value=make_manhattan_distance_heuristic(gw),
        randomize_action_order=True,
        seed=42
    )
    res = planner.plan_on(gw)
    assert [(s['x'], s['y']) for s in res.path] == soln2
