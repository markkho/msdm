import numpy as np
from msdm.algorithms.policyiteration import PolicyIteration
from msdm.core.mdp import TabularMarkovDecisionProcess, QuickMDP
from msdm.core.distributions import DictDistribution

from msdm.tests.domains import AbsorbingStateTester, DeterministicCounter, DeterministicUnreachableCounter, GNTFig6_6, \
    GeometricCounter, VaryingActionNumber, DeadEndBandit, TiedPaths, LineWorld, \
    RussellNorvigGrid_Fig17_3, PositiveRewardCycle, RussellNorvigGrid, SlipperyMaze

test_mdps = [
    DeterministicCounter(3, discount_rate=1.0),
    DeterministicUnreachableCounter(3, discount_rate=.95),
    GeometricCounter(p=1/13, discount_rate=1.0),
    GeometricCounter(p=1/13, discount_rate=.95),
    GeometricCounter(p=1/13, discount_rate=.513),
    PositiveRewardCycle(discount_rate=.95),
    VaryingActionNumber(discount_rate=1.0),
    TiedPaths(discount_rate=1.0),
    TiedPaths(discount_rate=.99),
    RussellNorvigGrid_Fig17_3(discount_rate=1.0), 
    RussellNorvigGrid(
        discount_rate=.41192,
        slip_prob=.7140
    ),
    GNTFig6_6()
]

def test_MarkovDecisionProcess_reachable_states():
    line_world = QuickMDP(
        next_state_dist=lambda s, a: 
            DictDistribution({s + a: .1, s: .9}) if 0 <= (s + a) <= 5 else \
            DictDistribution({s: 1.0}),
        reward=lambda s, a, ns: 0,
        actions=(-1, 1),
        initial_state_dist=DictDistribution({0: 1, 10: 0}),
        is_absorbing=lambda s: s == -1
    )
    assert set(line_world.reachable_states()) == {0, 1, 2, 3, 4, 5}
    assert set(line_world.reachable_states(max_states=2)) == {0, 1}
    
    bounded_lineworld = QuickMDP(
        next_state_dist=lambda s, a: DictDistribution({s + a: .8, s - a: .1, s: .1}),
        reward=lambda s, a, ns: -1,
        actions=(-1, 1),
        initial_state_dist=DictDistribution({0: 1}),
        is_absorbing=lambda s: ((s <= -1) or (s >= 1))
    )
    reachable = bounded_lineworld.reachable_states(max_states=10)
    assert reachable == set([-1, 0, 1])

def test_TabularMarkovDecisionProcess_implicit_absorbing_state_detection():
    # action space and reward function leads to last being an absorbing state
    mdp = AbsorbingStateTester(discount_rate=1.0, last_reward=0, last_actions=(0, ))
    assert (mdp.absorbing_state_vec == np.array([False, False, True])).all()

    # reward function leads to last not being an absorbing state
    mdp = AbsorbingStateTester(discount_rate=1.0, last_reward=1, last_actions=(0, ))
    assert (mdp.absorbing_state_vec == np.array([False, False, False])).all()
    # set absorbing state explicitly
    mdp = AbsorbingStateTester(
        discount_rate=1.0, last_reward=1, last_actions=(0, ),
        explicit_absorbing_flag=True
    )
    assert (mdp.absorbing_state_vec == np.array([False, False, True])).all()

    # action space leads to last not being an absorbing state
    mdp = AbsorbingStateTester(discount_rate=1.0, last_reward=0, last_actions=(-1, 0))
    assert (mdp.absorbing_state_vec == np.array([False, False, False])).all()
    # set absorbing state explicitly
    mdp = AbsorbingStateTester(
        discount_rate=1.0, last_reward=1, last_actions=(0, -1),
        explicit_absorbing_flag=True
    )
    assert (mdp.absorbing_state_vec == np.array([False, False, True])).all()

def test_TabularMarkovDecisionProcess_dead_end_detection():
    mdp = DeadEndBandit()
    assert mdp.dead_end_state_vec.any()

def test_TabularMarkovDecisionProcess_from_matrices():
    for mdp1 in test_mdps:
        mdp1 : TabularMarkovDecisionProcess
        print(mdp1)
        mdp2 = TabularMarkovDecisionProcess.from_matrices(
            state_list = mdp1.state_list,
            action_list = mdp1.action_list,
            initial_state_vec = mdp1.initial_state_vec,
            transition_matrix = mdp1.transition_matrix,
            action_matrix = mdp1.action_matrix,
            reward_matrix = mdp1.reward_matrix,
            absorbing_state_vec = mdp1.absorbing_state_vec,
            discount_rate = mdp1.discount_rate
        )
        pi1 = PolicyIteration().plan_on(mdp1)
        pi2 = PolicyIteration().plan_on(mdp2)

        assert mdp2.state_list == mdp1.state_list
        assert mdp2.action_list == mdp1.action_list
        assert (mdp2.initial_state_vec == mdp1.initial_state_vec).all()
        assert (mdp2.transition_matrix == mdp1.transition_matrix).all()
        assert (mdp2.action_matrix == mdp1.action_matrix).all()
        assert (mdp2.reward_matrix == mdp1.reward_matrix).all()
        assert mdp2.discount_rate == mdp1.discount_rate

        assert np.isclose(pi1.initial_value, pi2.initial_value)
        assert pi1.iterations == pi2.iterations
        assert pi1.converged == pi2.converged

def test_QuickMDP_equivalence():
    for mdp in test_mdps:
        mdp : TabularMarkovDecisionProcess
        quick_mdp = QuickMDP(
            discount_rate      = mdp.discount_rate,
            initial_state_dist = mdp.initial_state_dist,
            actions            = mdp.actions,
            next_state_dist    = mdp.next_state_dist,
            is_absorbing        = mdp.is_absorbing,
            reward             = mdp.reward,
        )
        assert mdp.initial_state_dist().isclose(quick_mdp.initial_state_dist())
        for s in mdp.state_list:
            assert set(mdp.actions(s)) == set(quick_mdp.actions(s))
            assert mdp.is_absorbing(s) == quick_mdp.is_absorbing(s)
            for a in mdp.actions(s):
                assert mdp.next_state_dist(s, a).isclose(quick_mdp.next_state_dist(s, a))
