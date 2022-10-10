import numpy as np 

from msdm.algorithms.valueiteration_new import ValueIteration
from msdm.tests.domains import DeterministicCounter, DeterministicUnreachableCounter, GNTFig6_6, \
    GeometricCounter, VaryingActionNumber, DeadEndBandit, TiedPaths, \
    RussellNorvigGrid_Fig17_3, PositiveRewardCycle

def test_tabular_planner_correctness():
    planners = [
        ValueIteration(max_iterations=1000, max_residual=1e-5, _version="dict"),
        ValueIteration(max_iterations=1000, max_residual=1e-5, _version="vectorized")
    ]
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
        RussellNorvigGrid_Fig17_3()
    ]

    for pl in planners:
        for mdp in test_mdps:
            print(f"Testing {pl} on {mdp}")
            result = pl.plan_on(mdp)
            optimal_policy = mdp.optimal_policy()
            optimal_state_value = mdp.optimal_state_value()
            assert len(optimal_policy) == len(result.policy)
            assert len(optimal_state_value) == len(result.state_value) == len(mdp.state_list)
            for s in optimal_policy:
                assert optimal_policy[s].isclose(result.policy[s])
                assert np.isclose(optimal_state_value[s], result.state_value[s], atol=1e-3)