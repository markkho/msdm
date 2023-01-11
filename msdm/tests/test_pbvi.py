from msdm.algorithms import LAOStar, PointBasedValueIteration, QMDP
from msdm.domains.tiger import Tiger
from msdm.domains.heavenorhell import HeavenOrHell
from msdm.core.pomdp import BeliefMDP

def compare_lao_and_pbvi(pomdp):
    # LAO* can be run on the belief-state MDP directly
    # to get an optimal policy. But, it requires that the
    # heuristic be admissible.
    bmdp = BeliefMDP(pomdp)

    lao = LAOStar(
        seed=124977,
        heuristic=lambda s: pomdp.reward_matrix.max(),
        max_lao_star_iterations=100,
        dynamic_programming_iterations=100,
    )
    lao_res = lao.plan_on(bmdp)
    assert lao_res.converged
    pbvi_res = PointBasedValueIteration(
        min_belief_expansions=5,
        max_belief_expansions=100
    ).plan_on(pomdp)

    # compare policies on visited states
    bmdp_states = lao_res.solution_graph.states_to_nodes.keys()
    for b in bmdp_states:
        if bmdp.is_absorbing(b):
            continue
        pbvi_pol = set(pbvi_res.policy.action_dist(b).support)
        lao_pol = set(lao_res.policy.action_dist(b).support)
        # lao* doesn't always preserve symmatry, so we test if it's actions are a subset
        assert lao_pol <= pbvi_pol

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

def test_pbvi_qmdp():
    # PBVI will realize it needs to go to the c square
    # to learn, whereas QMDP assumes it will know where
    # to go after the next step
    hh = HeavenOrHell(
        coherence=.6,
        grid=
            """
            g.h
            .sc
            h.g
            """,
        discount_rate=.95,
        heaven_reward=50,
        hell_reward=-50,
    )
    pbvi_res = PointBasedValueIteration(
        min_belief_expansions=5,
        max_belief_expansions=100
    ).plan_on(hh)
    qmdp_res = QMDP().plan_on(hh)
    assert list(qmdp_res.policy.action_dist(qmdp_res.policy.initial_agentstate()).probs) == [.25, .25, .25, .25]
    assert list(pbvi_res.policy.action_dist(pbvi_res.policy.initial_agentstate()).probs) == [1]
