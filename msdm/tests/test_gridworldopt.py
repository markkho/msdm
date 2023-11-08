import numpy as np
from msdm.domains.gridworldopt.mdp import map_to_xy_array, xy_array_to_map
from msdm.domains.gridworldopt.mdp import GridWorld
from msdm.core.distributions import DictDistribution

params = dict(
    feature_map=[
        "@..x$",
        "##.xb",
        "@yx.b",
    ],
    feature_rewards={
        'x': -5,
        'y': -10,
        "$": 9.12
    },
    absorbing_features = ("$",),
    wall_features= ("#",),
    initial_features = ("@",),
    step_cost = -1,
    wall_bump_cost  = -10,
    stay_prob  = 0.01,
    left_slip_prob  = 0.1,
    right_slip_prob  = 0.1,
    back_slip_prob  = 0.04,
    wait_action  = True,
    discount_rate = .99,
)

def test_GridWorld_methods():
    gw = GridWorld(**params)
    assert len(gw.reachable_states()) == 13
    assert sum([gw.is_absorbing(s) for s in gw.state_list]) == 1
    assert gw.initial_state_dist().isclose(
        DictDistribution([
            ((0, 0), .5),
            ((0, 2), .5),
        ])
    )

def test_GridWorld_rewards():
    gw = GridWorld(**params)
    # current implementation just gives expected reward for an action
    sans_exp_r = [
        ((0, 0), (1, 0), (1, 0), -1 + (-10)*.75 + (-10*.24)),
        ((0, 0), (1, 0), (0, 0), -1 + (-10)*.75 + (-10*.24)),
        ((2, 1), (1, 0), (2, 1), -1 + (-10)*(.04) + (-5)*(.75 + .1)),
    ]
    for s, a, ns, r in sans_exp_r:
        assert gw.reward(s, a, ns) == r, (s, a, ns, r, gw.reward(s, a, ns))

def test_GridWorld_transitions():
    gw = GridWorld(**params)
    sans_exp_prob = [
        ((0, 0), (1, 0), (0, 0), .25),
        ((0, 0), (1, 0), (1, 0), .75),
        ((0, 0), (-1, 0), (0, 0), .96),
        ((0, 0), (-1, 0), (1, 0), .04),
        ((3, 1), (0, 1), (3, 2), .75),
        ((3, 1), (0, 1), (3, 0), .04),
        ((3, 1), (0, 1), (3, 1), .01),
        ((3, 1), (0, 1), (2, 1), .1),
        ((3, 1), (0, 1), (4, 1), .1),
    ]
    for s, a, ns, p in sans_exp_prob:
        assert gw.next_state_dist(s, a)[ns] == p, (s, a, ns, p, gw.next_state_dist(s, a)[ns])
    assert np.all(np.isclose(gw.transition_matrix.sum(-1), 1))

def test_feature_map_to_xy_array():
    arr = GridWorld(**{
        **params,
    }).feature_map_array
    xy_arr = map_to_xy_array(arr)
    exp = {
        (0, 0): '@',
        (0, 1): '#',
        (0, 2): '@',
        (1, 0): 'y',
        (1, 1): '#',
        (1, 2): '.',
        (2, 0): 'x',
        (2, 1): '.',
        (2, 2): '.',
        (3, 0): '.',
        (3, 1): 'x',
        (3, 2): 'x',
    }
    for (x, y), c in exp.items():
        assert xy_arr[x, y] == c, (x, y, c, xy_arr[x, y])
    assert np.all(xy_array_to_map(xy_arr) == arr)