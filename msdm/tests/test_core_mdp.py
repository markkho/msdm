import numpy as np
from frozendict import frozendict
from msdm.core.problemclasses.mdp.quickmdp import QuickMDP
from msdm.domains import GridWorld
from msdm.algorithms import PolicyIteration, ValueIteration
from msdm.core.problemclasses.mdp import TabularPolicy, TabularMarkovDecisionProcess, QuickTabularMDP
from msdm.core.distributions import DictDistribution

from msdm.tests.domains import DeterministicCounter, DeterministicUnreachableCounter, GNTFig6_6, \
    GeometricCounter, VaryingActionNumber, DeadEndBandit, TiedPaths, \
    RussellNorvigGrid_Fig17_3, PositiveRewardCycle, RussellNorvigGrid

test_mdps = [
    DeterministicCounter(3, discount_rate=1.0),
    DeterministicUnreachableCounter(3, discount_rate=.95),
    GeometricCounter(p=1/13, discount_rate=1.0),
    GeometricCounter(p=1/13, discount_rate=.95),
    GeometricCounter(p=1/13, discount_rate=.513),
    PositiveRewardCycle(),
    VaryingActionNumber(),
    DeadEndBandit(),
    TiedPaths(discount_rate=1.0),
    TiedPaths(discount_rate=.99),
    RussellNorvigGrid_Fig17_3(), 
    RussellNorvigGrid(
        discount_rate=.41192,
        slip_prob=.7140
    )

]

def test_MarkovDecisionProcess_reachable_states():
    line_world = QuickMDP(
        next_state_dist=lambda s, a: 
            DictDistribution({s + a: .1, s: .9}) if 0 <= (s + a) <= 5 else \
            DictDistribution({s: 1.0}),
        reward=lambda s, a, ns: 0,
        actions=(-1, 1),
        initial_state_dist=DictDistribution({0: 1, 10: 0}),
        is_terminal=lambda s: s == -1
    )
    assert set(line_world.reachable_states()) == {0, 1, 2, 3, 4, 5}
    assert set(line_world.reachable_states(max_states=2)) == {0, 1}

def test_QuickMDP_equivalence():
    from msdm.core.problemclasses.mdp.quickmdp import QuickMDP
    from msdm.core.problemclasses.mdp.quicktabularmdp import QuickTabularMDP 

    for mdp in test_mdps:
        mdp :TabularMarkovDecisionProcess
        quick_mdp = QuickMDP(
            discount_rate      = mdp.discount_rate,
            initial_state_dist = mdp.initial_state_dist,
            actions            = mdp.actions,
            next_state_dist    = mdp.next_state_dist,
            is_terminal        = mdp.is_terminal,
            reward             = mdp.reward,
        )
        assert mdp.initial_state_dist().isclose(quick_mdp.initial_state_dist())
        for s in mdp.state_list:
            assert set(mdp.actions(s)) == set(quick_mdp.actions(s))
            assert mdp.is_terminal(s) == quick_mdp.is_terminal(s)
            for a in mdp.actions(s):
                assert mdp.next_state_dist(s, a).isclose(quick_mdp.next_state_dist(s, a))

def test_TabularMarkovDecisionProcess_from_matrices():
    from msdm.algorithms.policyiteration_new import PolicyIteration

    for mdp1 in test_mdps:
        print(mdp1)
        mdp2 = TabularMarkovDecisionProcess.from_matrices(
            state_list = mdp1.state_list,
            action_list = mdp1.action_list,
            initial_state_vec = mdp1.initial_state_vec,
            transition_matrix = mdp1.transition_matrix,
            action_matrix = mdp1.action_matrix,
            reward_matrix = mdp1.reward_matrix,
            nonterminal_state_vec = mdp1.nonterminal_state_vec,
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
        assert (mdp2.nonterminal_state_vec == mdp1.nonterminal_state_vec).all()
        assert mdp2.discount_rate == mdp1.discount_rate

        assert np.isclose(pi1.initial_value, pi2.initial_value)
        assert pi1.iterations == pi2.iterations
        assert pi1.converged == pi2.converged

def test_tabularpolicy_softmax():
    mdp = QuickTabularMDP(
        next_state_dist=lambda s, a: DictDistribution({s + a: .9, s: .1}) if 0 <= s+a < 6 else DictDistribution({s: 1}),
        reward=lambda s, a, ns: -1,
        actions=(-1, 1),
        initial_state_dist=DictDistribution({0: 1}),
        is_terminal=lambda s: s == 5
    )
    res = PolicyIteration().plan_on(mdp)
    softhard_pi = TabularPolicy.from_q_matrix(mdp.state_list, mdp.action_list, res._qvaluemat, inverse_temperature=1e10)
    hard_v0 = res.policy.evaluate_on(mdp).initial_value
    softhard_v0 = softhard_pi.evaluate_on(mdp).initial_value
    assert np.isclose(hard_v0, softhard_v0)
