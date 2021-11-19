import numpy as np
import random

from msdm.algorithms.laostar import LAOStar, ExplicitStateGraph
from msdm.domains import GridWorld
from msdm.tests.domains import make_russell_norvig_grid
from msdm.algorithms import PolicyIteration, ValueIteration
from msdm.tests.domains import Counter

def test_laostar_correctness():
    VALUE_TOLERANCE = 1e-8
    mdps = [
        GridWorld(
            tile_array=[
                ".............",
                "....######...",
                "....#....#...",
                "#.###..#.#...",
                "....#..#.#..g",
                "....#..#.#...",
                ".......#.####",
                "######.#.#...",
                "s....#.#.#...",
                "...#...#.....",
            ],
            discount_rate=.99
        ),
        make_russell_norvig_grid(
            discount_rate=.95,
            slip_prob=.8
        ),
        make_russell_norvig_grid(
            discount_rate=1-1e-8,
            slip_prob=.8
        ),
        make_russell_norvig_grid(
            discount_rate=.8,
            slip_prob=.6
        )
    ]
    rng = random.Random(124391)
    for mdp in mdps:
        vmax = mdp.reward_matrix.max()/(1 - mdp.discount_rate)
        heuristic = lambda s: vmax
        pi_res = PolicyIteration().plan_on(mdp)
        for seed in [rng.randint(0, int(1e20)) for _ in range(2)]:
            lao_res = LAOStar(
                seed=seed,
                heuristic=heuristic
            ).plan_on(mdp)
            assert np.isclose(lao_res.initial_value, pi_res.initial_value, atol=VALUE_TOLERANCE), (lao_res.initial_value, pi_res.initial_value)

def test_explicit_state_graph_expansion_and_dynamic_programming():
    gw = GridWorld(
        tile_array=[
            ".........g",
            "..........",
            "..########",
            "..........",
            "#######...",
            "s.........",
        ],
        discount_rate=.99
    )
    explicit_graph = ExplicitStateGraph(
        mdp=gw,
        heuristic=lambda s: 0,
        rng=random.Random(1234197),
        dynamic_programming_iterations=100,
    )
    # this procedure should encounter all reachable states
    explicit_graph.expand_while(lambda s: True)
    assert len(explicit_graph.state_list) == len(gw.reachable_states())

    # the solution graph should be solved
    solution_graph = explicit_graph.solution_graph()
    assert solution_graph.is_solved()

    # doing dynamic programming over all reachable states should give the same
    # value as policy iteration
    explicit_graph.dynamic_programming(list(explicit_graph.states_to_nodes.values()))
    initial_value = explicit_graph.states_to_nodes[gw.initial_state_dist().support[0]].value
    pi_res = PolicyIteration().plan_on(gw)
    vi_res = ValueIteration().plan_on(gw)
    assert np.isclose(initial_value, pi_res.initial_value)
    assert np.isclose(initial_value, vi_res.initial_value)

def test_explicit_state_graph_with_heuristic():
    # the manhattan distance heuristic should lead to a correct value function
    # beyond y = 3
    gw = GridWorld(
        tile_array=[
            "g.........",
            "..........",
            "..........",
            "#########.",
            "...#...#..",
            "s#...#....",
        ],
        discount_rate=.99
    )
    goal = gw.feature_locations['g'][0]
    explicit_graph = ExplicitStateGraph(
        mdp=gw,
        heuristic=lambda s: -(abs(s['x'] - goal['x']) + abs(s['y'] - goal['y'])),
        rng=random,
        dynamic_programming_iterations=100,
    )
    explicit_graph.expand_while(lambda s: s['y'] <= 3)
    expanded_nodes = [n for s, n in explicit_graph.states_to_nodes.items() if n.expanded]
    explicit_graph.dynamic_programming(expanded_nodes)
    vi_res = ValueIteration().plan_on(gw)
    initial_value = explicit_graph.states_to_nodes[gw.initial_state_dist().support[0]].value
    assert np.isclose(initial_value, vi_res.initial_value, atol=1e-10)

    # the solution graph should NOT be solved since there are non-terminal tips
    solution_graph = explicit_graph.solution_graph()
    assert not solution_graph.is_solved()

def test_trivial_solution():
    lao = LAOStar(
        heuristic = lambda s : 0,
        seed=42
    )
    # Normal
    mdp = Counter(3, initial_state=0)
    res = lao.plan_on(mdp)
    assert res.solution_graph.states_to_nodes[mdp.initial_state()]['value'] == -3
    assert res.policy.run_on(mdp).action_traj == (+1, +1, +1)

    # No-op task. Now we start at 3, so value should be 0 there
    mdp = Counter(3, initial_state=3)
    res = lao.plan_on(mdp)
    print(res.solution_graph.states_to_nodes)
    assert res.solution_graph.states_to_nodes[mdp.initial_state()]['value'] == 0
    assert res.policy.run_on(mdp).action_traj == ()
