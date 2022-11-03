import numpy as np
import warnings
from msdm.algorithms.multichainpolicyiteration import MultichainPolicyIteration 

from msdm.algorithms.valueiteration import ValueIteration
from msdm.algorithms.policyiteration import PolicyIteration 
from msdm.core.algorithmclasses import Plans
from msdm.tests.domains import DeterministicCounter, DeterministicUnreachableCounter, GNTFig6_6, \
    GeometricCounter, TestDomain, VaryingActionNumber, DeadEndBandit, TiedPaths, \
    RussellNorvigGrid_Fig17_3, PositiveRewardCycle, Puterman_Example_9_1_1

discounted_mdps = [
    PositiveRewardCycle(discount_rate=.95),
    DeterministicUnreachableCounter(3, discount_rate=.95),
    DeterministicCounter(3, discount_rate=.95),
    GeometricCounter(p=1/13, discount_rate=.95),
    GeometricCounter(p=1/13, discount_rate=.513),
    TiedPaths(discount_rate=.99),
    Puterman_Example_9_1_1(discount_rate=.9312)
]

stochastic_shortest_path_mdps =[
    VaryingActionNumber(discount_rate=1.0),
    DeterministicCounter(3, discount_rate=1.0),
    GeometricCounter(p=1/13, discount_rate=1.0),
    TiedPaths(discount_rate=1.0),
]

# An ergodic or recurrent MDP is one where the transition matrix corresponding to every 
# (deterministic stationary) policy has a single recurrent class. 
ergodic_undiscounted_mdps = [
    RussellNorvigGrid_Fig17_3(discount_rate=1.0),
]
safe_mdps = \
    discounted_mdps +\
    stochastic_shortest_path_mdps +\
    ergodic_undiscounted_mdps

non_terminating_mdps = [
    Puterman_Example_9_1_1(discount_rate=1.0),
    DeterministicUnreachableCounter(3, discount_rate=1.0),
]
# deadend_mdps = [
#     DeadEndBandit(),
# ]

def _test_tabular_planner_correctness(pl: Plans, test_mdps):
    for mdp in test_mdps:
        assert isinstance(mdp, TestDomain)
        print(f"Testing {pl} on {mdp}")
        result = pl.plan_on(mdp)
        optimal_policy = mdp.optimal_policy()
        optimal_state_value = mdp.optimal_state_value()
        assert result.converged
        assert len(optimal_policy) == len(result.policy)
        assert len(optimal_state_value) == len(result.state_value) == len(mdp.state_list)
        for s in optimal_policy:
            assert optimal_policy[s].isclose(result.policy[s])
            assert np.isclose(optimal_state_value[s], result.state_value[s], atol=1e-3)
            if hasattr(mdp, "optimal_state_gain") and hasattr(result, "state_gain"):
                assert np.isclose(mdp.optimal_state_gain()[s], result.state_gain[s], atol=1e-3)

def test_ValueIteration_dict_correctness():
    _test_tabular_planner_correctness(
        ValueIteration(max_iterations=1000, max_residual=1e-5, _version="dict"),
        test_mdps=safe_mdps
    )

def test_ValueIteration_dict_with_recurrent_states():
    vi = ValueIteration(max_iterations=100, undefined_value=0, _version="dict")
    recurrent_mdp = DeterministicUnreachableCounter(3, discount_rate=1.0)
    vi_res = vi.plan_on(recurrent_mdp)
    for s, v in recurrent_mdp.optimal_state_value().items():
        assert np.isclose(vi_res.state_value[s], v)

def test_ValueIteration_vec_correctness():
    _test_tabular_planner_correctness(
        ValueIteration(max_iterations=1000, max_residual=1e-5, _version="vectorized"),
        test_mdps=safe_mdps
    )

def test_ValueIteration_vec_with_recurrent_states():
    vi = ValueIteration(max_iterations=100, undefined_value=0, _version="vectorized")
    recurrent_mdp = DeterministicUnreachableCounter(3, discount_rate=1.0)
    vi_res = vi.plan_on(recurrent_mdp)
    for s, v in recurrent_mdp.optimal_state_value().items():
        assert np.isclose(vi_res.state_value[s], v)

def test_PolicyIteration_correctness():
    _test_tabular_planner_correctness(
        PolicyIteration(max_iterations=1000, undefined_value=float('-inf')),
        test_mdps=safe_mdps
    )

def test_PolicyIteration_with_recurrent_states():
    pi = PolicyIteration(max_iterations=1000, undefined_value=0)
    recurrent_mdp = DeterministicUnreachableCounter(3, discount_rate=1.0)
    with warnings.catch_warnings(record=True) as w:
        pi_res = pi.plan_on(recurrent_mdp)
        assert len(w) == 1
        assert 'MDP contains states that never reach an absorbing state' in str(w[0])
    for s, v in recurrent_mdp.optimal_state_value().items():
        assert np.isclose(pi_res.state_value[s], v)

def test_MultiChainPolicyIteration_vec_correctness():
    _test_tabular_planner_correctness(
        MultichainPolicyIteration(max_iterations=1000),
        test_mdps=safe_mdps+non_terminating_mdps
    )