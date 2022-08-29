from msdm.algorithms import RMAX
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
