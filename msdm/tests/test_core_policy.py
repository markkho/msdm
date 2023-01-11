import random
import numpy as np
from msdm.tests.domains import GeometricCounter, DeterministicCounter, \
    DeterministicUnreachableCounter, PositiveRewardCycle, VaryingActionNumber, TiedPaths, \
    LineWorld
from msdm.core.distributions import DictDistribution
from msdm.core.mdp import Policy, FunctionalPolicy

test_mdps = [
    DeterministicCounter(3, discount_rate=1.0),
    DeterministicUnreachableCounter(3, discount_rate=.95),
    GeometricCounter(p=1/4, discount_rate=.95),
    PositiveRewardCycle(),
    VaryingActionNumber(),
    TiedPaths(discount_rate=1.0),
]

def test_Policy_run_on_seed():
    for mdp in test_mdps:
        policy = FunctionalPolicy(lambda s: DictDistribution.uniform(mdp.actions(s)))
        seed = random.randint(0, int(1e8))
        sim_params = dict(
            mdp=mdp,
            initial_state=None,
            max_steps=100,
        )
        res1 = policy.run_on(
            **sim_params,
            rng=random.Random(seed),
        )
        res2 = policy.run_on(
            **sim_params,
            rng=random.Random(seed),
        )
        assert res1 == res2
        
def test_Policy_to_tabular():
    # convert to a tabular representation and confirm its equivalent
    for mdp in test_mdps:
        policy = FunctionalPolicy(lambda s: DictDistribution.uniform(mdp.actions(s)))
        tb_policy = policy.to_tabular(mdp.state_list, mdp.action_list)
        for s in mdp.state_list:
            assert tb_policy[s].isclose(policy.action_dist(s))
    
def test_TabularPolicy_Policy_evaluate_on():
    rng = random.Random(123999)
    for mdp in test_mdps:
        policy = FunctionalPolicy(
            lambda s: 
                mdp.optimal_policy()[s]*.9 | \
                DictDistribution.uniform(mdp.actions(s))*.1
        )
        tb_policy = policy.to_tabular(mdp.state_list, mdp.action_list)
        tab_eval = tb_policy.evaluate_on(mdp)
        sim_eval_few_err = []
        sim_eval_many_err = []
        for _ in range(10):
            sim_eval_few = policy.evaluate_on(mdp, n_simulations=4, max_steps=10, rng=rng)
            sim_eval_few_err.append(abs(sim_eval_few.initial_value - tab_eval.initial_value))
            sim_eval_many = policy.evaluate_on(mdp, n_simulations=50, max_steps=10, rng=rng)
            sim_eval_many_err.append(abs(sim_eval_many.initial_value - tab_eval.initial_value))
        assert np.mean(sim_eval_few_err) > np.mean(sim_eval_many_err), (np.mean(sim_eval_few_err), np.mean(sim_eval_many_err))

def test_TabularPolicy_undiscounted_negative_evaluate_on():
    lw_inf = LineWorld(line="s..#...#s.g", discount_rate=1.0)
    lw_finite = LineWorld(line="...#...#s.g", discount_rate=1.0)

    pol = FunctionalPolicy(lambda s: DictDistribution({1: 1})).to_tabular(
        state_list=lw_inf.state_list, action_list=lw_inf.action_list
    )

    res_inf = pol.evaluate_on(lw_inf)
    res_finite = pol.evaluate_on(lw_finite)
    assert res_inf.state_value.equivalent_to(res_finite.state_value)
    assert res_inf.action_value.equivalent_to(res_finite.action_value)
    assert res_inf.initial_value == float('-inf')
    assert res_finite.initial_value == -2
    assert np.array(res_inf.state_occupancy).sum() == float('inf')
    assert np.array(res_finite.state_occupancy).sum() < float('inf')
