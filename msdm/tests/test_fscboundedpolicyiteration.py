from msdm.domains.tiger import Tiger
from msdm.algorithms.fscboundedpolicyiteration import FSCBoundedPolicyIteration, improve_node_cvxpy
import numpy as np
import random

def test_fsc_bpi_tiger_cvxpy():
    '''
    The cvxpy-based version is a bit slower, so we have to fine-tune the parameters much more.
    '''
    pomdp = Tiger(coherence=0.95, discount_rate=0.95)

    # Sensitive to seed choice
    result = FSCBoundedPolicyIteration(controller_state_count=2, iterations=6, seed=47, improve_node_fn=improve_node_cvxpy).train_on(pomdp)
    assert result.value > 0

    # Also sensitive to seed choice, but is good at avoiding tiger.
    for s in result.policy.run_on(pomdp, max_steps=200, rng=random.Random(1337)):
        assert s.reward != -100

    # Check that this matches our default implementation.
    result2 = FSCBoundedPolicyIteration(controller_state_count=2, iterations=6, seed=47).train_on(pomdp)
    assert np.isclose(result.value, result2.value)
    assert np.allclose(result.policy.action_strategy, result2.policy.action_strategy)
    assert np.allclose(result.policy.observation_strategy, result2.policy.observation_strategy)

def test_fsc_bpi_tiger():
    pomdp = Tiger(coherence=0.95, discount_rate=0.95)

    b = FSCBoundedPolicyIteration(controller_state_count=2, iterations=20, seed=42)
    result = b.train_on(pomdp)
    assert result.value > 0

    for s in result.policy.run_on(pomdp, max_steps=1000, rng=random.Random(42)):
        assert s.reward != -100
