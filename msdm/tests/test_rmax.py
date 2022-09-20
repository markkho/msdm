import numpy as np
from msdm.algorithms import RMAX, PolicyIteration
from msdm.tests.domains import make_russell_norvig_grid


def test_rmax():
    gw = make_russell_norvig_grid(discount_rate=.95, slip_prob=0.8)

    rmax = RMAX(
        episodes=100,
        rmax=1,
        num_transition_samples=3,
        seed=0,
    )
    res = rmax.train_on(gw)
    pi_res = PolicyIteration().plan_on(gw)

    for s, a_qval in res.q_values.items():
        for a, qval in a_qval.items():
            exp_qval = pi_res.actionvaluefunc[s][a]
            assert np.isclose(exp_qval, qval), f"{s}, {a}, actual: {qval}, expected: {exp_qval}"