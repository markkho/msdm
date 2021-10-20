import numpy as np
from msdm.algorithms import QLearning, DoubleQLearning, SARSA, ExpectedSARSA
from msdm.tests.domains import make_russell_norvig_grid
from msdm.domains.cliffwalking import CliffWalking
from msdm.domains import GridWorld

def _test_tdlearner(Learner):
    gw = make_russell_norvig_grid(discount_rate=.95, slip_prob=0.8)

    # test reproducibility
    params = dict(
        seed = 12345,
        episodes=10,
        rand_choose=.0,
        step_size=.5,
        softmax_temp=0.01
    )
    ql1 = Learner(**params).train_on(gw)
    ql2 = Learner(**params).train_on(gw)
    ql3 = Learner(**{**params, 'seed': 54321}).train_on(gw)
    assert id(ql1) != id(ql2)
    assert ql1.event_listener_results.episode_rewards == ql2.event_listener_results.episode_rewards
    assert ql1.event_listener_results.episode_rewards != ql3.event_listener_results.episode_rewards

    # test simple learning set up works
    gw = GridWorld(
        tile_array=["s.g"],
        discount_rate=.99,
    )
    ql4 = Learner(**{**params, 'episodes': 1}).train_on(gw)
    ql5 = Learner(**{**params, 'episodes': 60}).train_on(gw)
    v0_4 = ql4.policy.evaluate_on(gw).initial_value
    v0_5 = ql5.policy.evaluate_on(gw).initial_value
    assert v0_4 < v0_5, Learner

def test_tdlearners():
    for Learner in [QLearning, DoubleQLearning, SARSA, ExpectedSARSA]:
        _test_tdlearner(Learner)

def test_td_algs():
    g = CliffWalking()
    ql = QLearning(rand_choose=.1, episodes=400, seed=1239123)
    qres = ql.train_on(g)
    qql = DoubleQLearning(rand_choose=.1, episodes=400, seed=1233)
    qqres = qql.train_on(g)
    sl = SARSA(rand_choose=.1, episodes=400, seed=239123)
    sres = sl.train_on(g)
    esl = ExpectedSARSA(rand_choose=.1, episodes=400, seed=12391)
    esres = esl.train_on(g)

    qr = np.mean(qres.event_listener_results.episode_rewards[-100:])
    qqr = np.mean(qqres.event_listener_results.episode_rewards[-100:])
    sr = np.mean(sres.event_listener_results.episode_rewards[-100:])
    esr = np.mean(esres.event_listener_results.episode_rewards[-100:])
    assert qr < qqr
    assert qqr < sr
    assert qqr < esr
