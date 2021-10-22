from msdm.algorithms import LAOStar, PointBasedValueIteration
from msdm.domains.tiger import Tiger
from msdm.domains.heavenorhell import HeavenOrHell
from msdm.core.problemclasses.pomdp import BeliefMDP

def compare_lao_and_pbvi(pomdp):
    # LAO* can be run on the belief-state MDP directly
    # to get an optimal policy. But, it requires that the
    # heuristic be admissible.
    bmdp = BeliefMDP(pomdp)
    lao = LAOStar(
        max_lao_iters=100,
        heuristic=lambda s: 0. if bmdp.is_terminal(s) else pomdp.reward_matrix.max(),
        show_progress=False
    )
    lao_res = lao.plan_on(bmdp)
    assert lao_res.converged
    pbvi_res = PointBasedValueIteration(
        min_belief_expansions=5,
        max_belief_expansions=100
    ).plan_on(pomdp)

    # compare policies on visited states
    bmdp_states = lao_res.sGraph.keys()
    for b in bmdp_states:
        if bmdp.is_terminal(b):
            continue
        pbvi_pol = sorted(pbvi_res.policy.action_dist(b).items())
        lao_pol = sorted(lao_res.policy.action_dist(b).items())
        if pbvi_pol != lao_pol:
            print(b)
            print(pbvi_pol)
            print(lao_pol)
            raise Exception("Policies don't match")

def test_pbvi_on_toy_domains():
    tiger = Tiger(
        coherence=.85,
        discount_rate=.85
    )
    hh = HeavenOrHell(
        coherence=.9,
        grid=
            """
            hsg
            #c#
            """,
        discount_rate=.9
    )
    compare_lao_and_pbvi(tiger)
    compare_lao_and_pbvi(hh)
