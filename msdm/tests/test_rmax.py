import numpy as np
from msdm.algorithms import RMAX
from msdm.tests.domains import make_russell_norvig_grid


def test_rmax():
    gw = make_russell_norvig_grid(discount_rate=.95, slip_prob=0.8)

    rmax = RMAX(
        episodes=200,
        rmax=1,
        num_transition_samples=10,
        seed=123907,
    )
    res = rmax.train_on(gw)

    ep_rewards = res.event_listener_results.episode_rewards
    assert np.mean(ep_rewards[:100]) < np.mean(ep_rewards[100:])