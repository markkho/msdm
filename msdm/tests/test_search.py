from msdm.algorithms import BreadthFirstSearch, AStarSearch, BidirectionalSearch
from msdm.core.mdp import MarkovDecisionProcess
from msdm.domains.gridworld.mdp import GridWorld, TERMINALSTATE
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

empty_gw = GridWorld(
    tile_array=[
        '....g',
        '.....',
        '.....',
        '.....',
        's....',
    ],
    feature_rewards={'g': 0},
    step_cost=-1,
    discount_rate=1.0
)

def test_bfs():
    res = BreadthFirstSearch().plan_on(gw)
    assert [(s['x'], s['y']) for s in res.path] == [(0, 2), (1, 2), (2, 2), (-1, -1)]

def test_deterministic_shortest_path():
    res = BreadthFirstSearch().plan_on(DeterministicCounter(3))
    assert res.path == [0, 1, 2, 3]

def make_manhattan_distance_heuristic(mdp : MarkovDecisionProcess):
    def manhattan_distance_heuristic(s):
        if mdp.is_absorbing(s):
            return 0
        goal = mdp.absorbing_states[0]
        dist = abs(s['x'] - goal['x']) + abs(s['y'] - goal['y'])
        return -dist
    return manhattan_distance_heuristic

def test_astarsearch():
    path_above = ((0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (-1, -1))
    path_below = ((0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (-1, -1))

    found = set()
    for seed in range(10):
        planner = AStarSearch(
            heuristic_value=make_manhattan_distance_heuristic(gw),
            randomize_action_order=False,
            seed=seed + 42,
        )
        res = planner.plan_on(gw)
        found.add(tuple([(s['x'], s['y']) for s in res.path]))
    # We deterministically select one of the two paths, because we tie-break with LIFO.
    # Depends on random seed.
    assert found == {path_below}

    found = set()
    for seed in range(10):
        planner = AStarSearch(
            heuristic_value=make_manhattan_distance_heuristic(gw),
            randomize_action_order=True,
            seed=seed + 42
        )
        res = planner.plan_on(gw)
        found.add(tuple([(s['x'], s['y']) for s in res.path]))
    # When action order is randomized, each of the optimal solutions are possible.
    # Depends on random seed, but likely.
    assert found == {path_above, path_below}

def test_astarsearch_tie_breaking():
    def _plan(kw):
        planner = AStarSearch(
            heuristic_value=make_manhattan_distance_heuristic(empty_gw),
            randomize_action_order=False,
            seed=kw.pop('seed', 42),
            **kw,
        )
        return planner.plan_on(empty_gw)

    def _check_path(res):
        assert res.path[-1] == TERMINALSTATE
        assert len(res.path) - 1 == path_len

    path_len = 9
    state_count = 25
    assert len(set(empty_gw.state_list) - {TERMINALSTATE}) == state_count

    # LIFO tie-breaking searches fewer states with equivalent cost + heuristic, because it
    # prioritizes more recently added entries to the queue. For the empty grid world we consider here,
    # the heuristic has no error, so cost + heuristic at all states is equal to the negative
    # value of the start state. LIFO ensures we focus on more recently visited states, which
    # means we only have to visit the states we wind up including in our path.
    res = _plan(dict(tie_breaking_strategy='lifo'))
    assert len(res.visited) == path_len
    assert res.visited == set(res.path[:-1])
    _check_path(res)

    # By contrast, FIFO is more thorough, prioritizing states added earlier. For this domain,
    # that means we wind up exploring the entire state space, since they have the same cost + heuristic.
    res = _plan(dict(tie_breaking_strategy='fifo'))
    assert len(res.visited) == state_count
    assert res.visited == set(empty_gw.state_list) - {TERMINALSTATE}
    _check_path(res)

    # Visits with random tie-breaking will be bounded by the above two extremes.
    for seed in range(100):
        res = _plan(dict(tie_breaking_strategy='random', seed=seed + 47283674))
        # This is guaranteed.
        assert path_len <= len(res.visited) <= state_count
        # This stricter bound will depend on the random seed, but is likely.
        assert path_len < len(res.visited) < state_count
        _check_path(res)

def test_bidirection_search():
    res = BidirectionalSearch().plan_on(gw)
    assert [(s['x'], s['y']) for s in res.path] == [(0, 2), (1, 2), (2, 2)]
    assert len(res.visited) == 7