import numpy as np
import pytest
import random

from msdm.algorithms.laostar import LAOStar, ExplicitStateGraph
from msdm.algorithms.policyiteration import PolicyIteration
from msdm.core.mdp import QuickTabularMDP
from msdm.core.distributions import DictDistribution
from msdm.algorithms.policyiteration import PolicyIteration
from msdm.algorithms.valueiteration import ValueIteration
from msdm.tests.domains import DeterministicCounter, make_russell_norvig_grid
from msdm.domains import GridWorld
from msdm.core.exceptions import SpecificationException

def test_laostar_random_action_ordering_flag():
    from frozendict import frozendict
    RIGHT = frozendict({'dx': 1, 'dy': 0})
    UP = frozendict({'dy': 1, 'dx': 0})
    class DefaultRightUpActionGridWorld(GridWorld):
        def actions(self, s):
            return (
                RIGHT,
                UP,
                frozendict({'dx': -1, 'dy': 0}),
                frozendict({'dy': -1, 'dx': 0}),
                frozendict({'dx': 0, 'dy': 0}),
            )

    gw_params = dict(
        tile_array=[
            "......g",
            ".......",
            ".......",
            "...s...",
            ".......",
            ".......",
            ".......",
        ]
    )
    gw = DefaultRightUpActionGridWorld(**gw_params)
    pi_res = PolicyIteration().plan_on(GridWorld(**gw_params, wall_features=""))

    nonrandomized_action_order_lao = LAOStar(
        heuristic=lambda s: pi_res.state_value[s],
        randomize_action_order=False,
        seed=1239123
    )
    nonrand_res = nonrandomized_action_order_lao.plan_on(gw)
    nonrand_action_traj = nonrand_res.policy.run_on(gw).action_traj
    nonrand_expected_traj = [RIGHT]*3 + [UP]*3
    nonrand_match = [taken == expected for taken, expected in zip(nonrand_action_traj, nonrand_expected_traj)]
    assert len(nonrand_match) == 6
    assert all(nonrand_match)
    nonrand_action_order = \
        [tuple(n.action_order) for n in nonrand_res.explicit_graph.states_to_nodes.values()]
    assert len(set(nonrand_action_order)) == 1

    randomized_action_order_lao = LAOStar(
        heuristic=lambda s: pi_res.state_value[s],
        randomize_action_order=True,
        seed=123123
    )
    rand_res = randomized_action_order_lao.plan_on(gw)
    rand_action_traj = rand_res.policy.run_on(gw).action_traj
    rand_match = [taken == expected for taken, expected in zip(rand_action_traj, nonrand_expected_traj)]
    assert len(rand_match) == 6
    assert not all(rand_match)
    rand_action_orders = \
        [tuple(n.action_order) for n in rand_res.explicit_graph.states_to_nodes.values()]
    assert len(set(rand_action_orders)) > 1


    nonrand_states_visited = list(nonrand_res.explicit_graph.states_to_nodes.keys())
    rand_states_visited = list(rand_res.explicit_graph.states_to_nodes.keys())
    assert set(nonrand_states_visited) != set(rand_states_visited)

def test_best_breadth_first_state_evaluation_order():
    def actions_CD_order(s):
        return {
            "A": ("A", "B"),
            "B": ("B", "CD"),
            "C": ("C", "CX"),
            "D": ("D", "DY")
        }.get(s, s)
    def actions_DC_order(s):
        return {
            "A": ("A", "B"),
            "B": ("B", "DC"),
            "C": ("C", "CX"),
            "D": ("D", "DY")
        }.get(s, s)
    def next_state_dist(s, a):
        # note we assume the order of the support will be preserved
        # when LAO* is building its explicit graph
        return DictDistribution.uniform(list(a))
    def reward(s, a, ns):
        return {
            "X": 1,
            "Y": 1
        }.get(ns, 0)
    def is_absorbing(s):
        return s in "XY"

    # these two MDPs are value-equivalent - only difference is the next state order
    # at state B, action CD is different
    mdp_CD = QuickTabularMDP(
        next_state_dist=next_state_dist,
        reward=reward,
        actions=actions_CD_order,
        initial_state="A",
        is_absorbing=is_absorbing,
        discount_rate=.99
    )
    mdp_DC = QuickTabularMDP(
        next_state_dist=next_state_dist,
        reward=reward,
        actions=actions_DC_order,
        initial_state="A",
        is_absorbing=is_absorbing,
        discount_rate=.99
    )
    pi_res_CD = PolicyIteration().plan_on(mdp_CD)
    pi_res_DC = PolicyIteration().plan_on(mdp_DC)
    assert pi_res_CD.initial_value == pi_res_DC.initial_value

    # D should always be expanded first since it is better
    # according to the heuristic
    lao_D_heuristic = LAOStar(
        heuristic=lambda s: {"C": 0, "D": 1}.get(s, .5),
        randomize_action_order=False,
        randomize_nextstate_order=False
    )
    lao_D_heuristic_rand = LAOStar(
        heuristic=lambda s: {"C": 0, "D": 1}.get(s, .5),
        randomize_action_order=False,
        randomize_nextstate_order=True,
        seed=12948
    )
    lao_res_D_heuristic_CD = lao_D_heuristic.plan_on(mdp_CD)
    lao_res_D_heuristic_DC = lao_D_heuristic.plan_on(mdp_DC)
    lao_res_D_heuristic_rand = lao_D_heuristic_rand.plan_on(mdp_CD)

    D_heuristic_order = ["A", "B", "D", "Y", "C", "X"]

    assert lao_res_D_heuristic_rand.explicit_graph.states_by_expandedorder() == D_heuristic_order
    assert lao_res_D_heuristic_CD.explicit_graph.states_by_expandedorder() == D_heuristic_order
    assert lao_res_D_heuristic_DC.explicit_graph.states_by_expandedorder() == D_heuristic_order


    # C or D should be evaluated first, depending on the mdp's order since the heuristic is tied
    lao_CD_heuristic = LAOStar(
        heuristic=lambda s: {"C": 1, "D": 1}.get(s, .5),
        randomize_action_order=False,
        randomize_nextstate_order=False
    )
    lao_res_CD_heuristic_CD = lao_CD_heuristic.plan_on(mdp_CD)
    lao_res_CD_heuristic_DC = lao_CD_heuristic.plan_on(mdp_DC)

    CD_order = lao_res_CD_heuristic_CD.explicit_graph.states_by_expandedorder()
    DC_order = lao_res_CD_heuristic_DC.explicit_graph.states_by_expandedorder()
    assert CD_order != DC_order

def test_laostar_duplicate_actions():
    def actions(s):
        return {
            "A": ("A", "B", "B"),
            "B": ("B", "X"),
        }.get(s, s)
    def next_state_dist(s, a):
        return DictDistribution.uniform(list(a))
    def reward(s, a, ns):
        return -1
    def is_absorbing(s):
        return s == "X"
    mdp = QuickTabularMDP(
        next_state_dist=next_state_dist,
        reward=reward,
        actions=actions,
        initial_state="A",
        is_absorbing=is_absorbing,
        discount_rate=.99
    )
    with pytest.raises(SpecificationException):
        LAOStar(heuristic=0).plan_on(mdp)

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
        randomize_action_order=True,
        randomize_nextstate_order=True,
        rng=random.Random(1234197),
        dynamic_programming_iterations=100,
    )
    # this procedure should encounter all reachable states
    explicit_graph.expand_while(lambda s: True)
    assert len(explicit_graph.states_to_nodes) == len(gw.reachable_states())

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
        randomize_action_order=True,
        randomize_nextstate_order=True,
        rng=random,
        dynamic_programming_iterations=100,
    )
    explicit_graph.expand_while(lambda s: s['y'] <= 3)
    expanded_nodes = [n for s, n in explicit_graph.states_to_nodes.items() if n.expanded]
    explicit_graph.dynamic_programming(expanded_nodes)
    vi_res = ValueIteration().plan_on(gw)
    initial_value = explicit_graph.states_to_nodes[gw.initial_state_dist().support[0]].value
    assert np.isclose(initial_value, vi_res.initial_value, atol=1e-10)

    # the solution graph should NOT be solved since there are non-absorbing tips
    solution_graph = explicit_graph.solution_graph()
    assert not solution_graph.is_solved()

def test_trivial_solution():
    lao = LAOStar(
        heuristic = lambda s : 0,
        seed=42
    )
    # Normal
    mdp = DeterministicCounter(3, initial_state=0)
    res = lao.plan_on(mdp)
    assert res.solution_graph.states_to_nodes[mdp.initial_state()]['value'] == -3
    assert res.policy.run_on(mdp).action_traj == (+1, +1, +1)

    # No-op task. Now we start at 3, so value should be 0 there
    mdp = DeterministicCounter(3, initial_state=3)
    res = lao.plan_on(mdp)
    print(res.solution_graph.states_to_nodes)
    assert res.solution_graph.states_to_nodes[mdp.initial_state()]['value'] == 0
    assert res.policy.run_on(mdp).action_traj == ()
