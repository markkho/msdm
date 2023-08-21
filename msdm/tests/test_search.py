import pytest

from msdm.algorithms import BreadthFirstSearch, AStarSearch
from msdm.core.mdp import MarkovDecisionProcess
from msdm.domains.gridworld.mdp import GridWorld, TERMINALSTATE
from msdm.tests.domains import DeterministicCounter, RomaniaSubsetAIMA


def _matching_states_from_tile_array(tile_array, matching):
    '''
    Returns states in `tile_array`, where the state's feature is in `matching`.
    '''
    return frozenset({
        s
        for s, feature in GridWorld(tile_array=tile_array).location_features.items()
        if feature in matching
    })

def _path_from_tile_array(tile_array, sequence):
    '''
    Returns a path of states specified in `tile_array`. State features along the path
    must start with sequence[0], proceeding until sequence[1] (inclusive).
    '''
    # First, index states by feature
    by_feature = {}
    for s, feature in GridWorld(tile_array=tile_array).location_features.items():
        by_feature.setdefault(feature, []).append(s)

    # Then, iterate over sequence to gather matches.
    rv = []
    sequence_start, sequence_end = sequence # end is inclusive
    for i in range(ord(sequence_end) - ord(sequence_start) + 1):
        curr_seq = chr(ord(sequence_start) + i)
        ss = by_feature[curr_seq]
        assert len(ss) == 1, f'Found {len(ss)} states matching for current sequence feature "{curr_seq}"'
        rv.append(ss[0])
    return tuple(rv)


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
gw.path_above = _path_from_tile_array([
    '234',
    '1h5',
    '0x6',
    '.h.',
    '...',
], ('0', '6')) + (TERMINALSTATE,)
gw.path_below = _path_from_tile_array([
    '...',
    '.h.',
    '0x6',
    '1h5',
    '234',
], ('0', '6')) + (TERMINALSTATE,)

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

bottleneck_gw = GridWorld(
    tile_array=[
        '...#g',
        '.#...',
        's..#.',
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
    found = set()
    for _ in range(10):
        planner = AStarSearch(
            heuristic_value=make_manhattan_distance_heuristic(gw),
            randomize_action_order=False,
        )
        res = planner.plan_on(gw)
        found.add(tuple(res.path))
    # We deterministically select one of the two paths, because we tie-break with LIFO.
    # Depends on random seed.
    assert found == {gw.path_below}

    found = set()
    for seed in range(10):
        planner = AStarSearch(
            heuristic_value=make_manhattan_distance_heuristic(gw),
            randomize_action_order=True,
            seed=seed + 42
        )
        res = planner.plan_on(gw)
        found.add(tuple(res.path))
    # When action order is randomized, each of the optimal solutions are possible.
    # Depends on random seed, but likely.
    assert found == {gw.path_above, gw.path_below}

def test_astarsearch_tie_breaking():
    def _plan(kw):
        planner = AStarSearch(
            heuristic_value=make_manhattan_distance_heuristic(empty_gw),
            randomize_action_order=True,
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

    visited_counts = set()
    # Visits with random tie-breaking will be bounded by the above two extremes.
    for seed in range(100):
        res = _plan(dict(tie_breaking_strategy='random', seed=seed + 47283674))
        assert path_len <= len(res.visited) <= state_count
        visited_counts.add(len(res.visited))
        _check_path(res)
    assert visited_counts == set(range(path_len, state_count + 1))

def test_astarsearch_revise_node():
    '''
    In this example, the heuristic might lead you to a bottleneck state via a suboptimal
    path. This test makes sure that the optimal path through the bottleneck is still considered.
    '''
    path = _path_from_tile_array([
        '...#6',
        '.#345',
        '012#.',
    ], ('0', '6')) + (TERMINALSTATE,)

    grid = [
        'CCB#A',
        'C#AAA',
        'AAA#.',
    ]

    expected = {
        (path, visited)
        for visited in [
            # This is the best case, we only consider path.
            _matching_states_from_tile_array(grid, 'A'),
            # In this case, we also consider one extra state, that looks good per the heuristic.
            _matching_states_from_tile_array(grid, 'AB'),
            # In this case, we also got derailed into considering the entire top of the problem.
            _matching_states_from_tile_array(grid, 'ABC'),
        ]
    }

    hv = make_manhattan_distance_heuristic(bottleneck_gw)
    results = set()
    for seed in range(100):
        res = AStarSearch(
            heuristic_value=hv,
            randomize_action_order=True,
            seed=seed + 58264359237,
        ).plan_on(bottleneck_gw)
        results.add((
            tuple(res.path),
            frozenset(res.visited),
        ))

    assert results == expected

def test_astarsearch_no_heuristic():
    mdp = RomaniaSubsetAIMA()
    result = AStarSearch(heuristic_value=lambda _: 0).plan_on(mdp)
    assert result.path == mdp.optimal_path()
    assert result.visited == {s for s in mdp.state_list if not mdp.is_absorbing(s)}, 'All non-absorbing states should be visited.'

def test_astarsearch_monotone_heuristic():
    base_hv = make_manhattan_distance_heuristic(gw)
    hv = lambda s: base_hv(s) * 1.1
    with pytest.raises(AssertionError) as err:
        res = AStarSearch(heuristic_value=hv).plan_on(gw)
    assert 'Heuristic is non-monotonic' in str(err)

    res = AStarSearch(heuristic_value=hv, assert_monotone_heuristic=False).plan_on(gw)
    assert tuple(res.path) == gw.path_below

def test_astarsearch_large_grid():
    gw = GridWorld(
        tile_array=[
            's#......#...',
            '..#########.',
            '....#.....#.',
            '#.#...##..#.',
            '....#...#.#.',
            '.#....#.#.#.',
            '...#..#....g'
        ],
        feature_rewards={'g': 0},
        absorbing_features='g',
        step_cost=-1,
        discount_rate=1.0,
    )
    hv = make_manhattan_distance_heuristic(gw)
    paths = set()
    for seed in range(100):
        res = AStarSearch(heuristic_value=hv, randomize_action_order=True, seed=823749285 + seed).plan_on(gw)
        paths.add(tuple(res.path))

    assert paths == {
        _path_from_tile_array([
            'a#......#...',
            'b.#########.',
            'cdef#.....#.',
            '#.#ghi##..#.',
            '....#jkl#.#.',
            '.#....#m#.#.',
            '...#..#nopqr'
        ], ('a', 'r')) + (TERMINALSTATE,),
        _path_from_tile_array([
            'a#......#...',
            'bc#########.',
            '.def#.....#.',
            '#.#ghi##..#.',
            '....#jkl#.#.',
            '.#....#m#.#.',
            '...#..#nopqr'
        ], ('a', 'r')) + (TERMINALSTATE,),
    }
