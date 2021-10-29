from msdm.domains.loadunload import LoadUnload
from msdm.domains.heavenorhell import HeavenOrHell
from msdm.algorithms.fscgradientascent import FSCGradientAscent
import random

def test_loadunload():
    pomdp = LoadUnload()
    res = FSCGradientAscent(learning_rate=1e-1, controller_state_count=2, iterations=100, seed=42).train_on(pomdp)
    t = res.policy.run_on(pomdp, max_steps=14 * 3, rng=random.Random(42))
    assert [s.state.location for s in t] == (list(range(8))+list(range(6, 0, -1))) * 3 + [0]
    assert [s.state.is_loaded for s in t] == ([False] * 7 + [True] * 7) * 3 + [False]

def test_hh_coherence1():
    grid="""
    hcg
    #s#
    """
    pomdp = HeavenOrHell(coherence=1, grid=grid)
    res = FSCGradientAscent(learning_rate=1e-1, controller_state_count=4, iterations=100, seed=42).train_on(pomdp)
    for s0, p in pomdp.initial_state_dist().items():
        t = res.policy.run_on(pomdp, initial_state=s0, rng=random.Random(42))
        assert [pomdp.loc_features[s.state.x, s.state.y] for s in t] == ['s', 'c', 'c', s0.heaven]
