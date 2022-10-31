import random
import numpy as np
from msdm.tests.domains import GeometricCounter, DeterministicCounter, \
    DeterministicUnreachableCounter, PositiveRewardCycle, VaryingActionNumber, TiedPaths
from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp.policy import Policy, FunctionalPolicy

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
