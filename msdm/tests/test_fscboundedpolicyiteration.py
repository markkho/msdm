from msdm.domains.tiger import Tiger
from msdm.algorithms.fscboundedpolicyiteration import FSCBoundedPolicyIteration
import random

def test_fsc_bpi_tiger():
    pomdp = Tiger(coherence=0.95, discount_rate=0.95)

    # Sensitive to seed choice
    b = FSCBoundedPolicyIteration(controller_state_count=2, iterations=6, seed=47)
    result = b.train_on(pomdp)
    assert result.value > 0

    # Also sensitive to seed choice, but is good at avoiding tiger.
    for s in result.policy.run_on(pomdp, max_steps=200, rng=random.Random(1337)):
        assert s.reward != -100
