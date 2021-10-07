from msdm.algorithms.qlearning import QLearning
from msdm.tests.domains import make_russell_norvig_grid
from msdm.domains import GridWorld

def test_qlearning():
    gw = make_russell_norvig_grid(discount_rate=.95, slip_prob=0.8)

    # test reproducibility
    params = dict(
        seed = 12345,
        episodes=10,
        rand_choose=.0,
        step_size=.5,
        softmax_temp=0.01
    )
    ql1 = QLearning(**params).train_on(gw)
    ql2 = QLearning(**params).train_on(gw)
    ql3 = QLearning(**{**params, 'seed': 54321}).train_on(gw)
    assert id(ql1) != id(ql2)
    assert ql1.episode_rewards == ql2.episode_rewards
    assert ql1.episode_rewards != ql3.episode_rewards

    # test simple learning set up works
    gw = GridWorld(
        tile_array=["s.g"],
        discount_rate=.99,
    )
    ql4 = QLearning(**{**params, 'episodes': 1}).train_on(gw)
    ql5 = QLearning(**{**params, 'episodes': 50}).train_on(gw)
    v0_4 = ql4.policy.evaluate_on(gw).initial_value
    v0_5 = ql5.policy.evaluate_on(gw).initial_value
    assert v0_4 < v0_5
