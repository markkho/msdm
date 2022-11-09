import copy
import pytest

from msdm.algorithms.lrtdp import LRTDP
from msdm.algorithms.valueiteration import ValueIteration
from msdm.core.mdp import MarkovDecisionProcess
from msdm.tests.domains import GNTFig6_6, DeterministicCounter
from msdm.domains import GridWorld
from msdm.domains.gridmdp.windygridworld import WindyGridWorld
from msdm.algorithms.lrtdp import LRTDPEventListener

def ensure_uniform(dist):
    '''
    Assumes the supplied distribution is uniform and returns values it assigns probability to.
    '''
    eps = 1e-4
    items = []
    prob = None
    for s in dist.support:
        if dist.prob(s) == 0:
            continue
        if prob is None:
            prob = dist.prob(s)
        assert abs(dist.prob(s) - prob) < eps
        items.append(s)
    return items

def deterministic(dist):
    '''
    Assumes the supplied distribution is deterministic and returns the deterministic value.
    '''
    for s in dist.support:
        if dist.prob(s) == 0:
            continue
        assert dist.prob(s) == 1
        return s

def _test_expected_error_bound(bellman_error_margin, optimal_res, test_res, mdp):
    """
    For an approximate dynamic programming algorithm initialized with an
    admissible heuristic that converges to a value function V and
    policy $\\pi$, we know that for all s:

    0 <= V(s) - V^*(s) <= \\eta x \\Phi^{\\pi}(s)

    where \\Phi^{\\pi} is the expected number of steps to reach an absorbing
    state from s under \\pi and \\eta is the error margin for the
    Bellman residual (see Ghallab, Nau & Traverso, pg 229).
    This function tests whether the expected error bound holds from the
    initial states of an MDP.
    """
    test_eval = test_res.policy.evaluate_on(mdp)
    test_occ = test_eval.state_occupancy  #occupancy from start state
    test_start_steps = sum(test_occ.values())
    value_diff = test_res.initial_value - optimal_res.initial_value
    within_bound = 0 <= value_diff <= bellman_error_margin*test_start_steps
    if not within_bound:
        raise OutOfExpectedErrorBound
    return bellman_error_margin*test_start_steps

class OutOfExpectedErrorBound(Exception):
    pass

def test_lrtdp_heuristics_on_stochastic_domain_multiple_discount_rates():
    _test_lrtdp_heuristics_on_stochastic_domain(discount_rate=1.0)
    _test_lrtdp_heuristics_on_stochastic_domain(discount_rate=.99)
    _test_lrtdp_heuristics_on_stochastic_domain(discount_rate=.9)

def _test_lrtdp_heuristics_on_stochastic_domain(discount_rate):
    bellman_error_margin = 1e-5
    wg = WindyGridWorld(
        grid="""
            ....$
            x^x<<
            x^x<<
            .^x<<
            x<<<<
            x<<<<
            x<<<<
            x<<<<
            x<<<<
            @....
        """,
        step_cost=-1,
        wall_bump_cost=-1,
        discount_rate=discount_rate,
        wind_probability=.5,
        feature_rewards={'x': -50, '$': 50}
    )
    vi_res = ValueIteration().plan_on(wg) #the ground truth
    lrtdp_res_admissible_shifted = LRTDP(
        heuristic=lambda s: vi_res.state_value[s] + 10,
        bellman_error_margin=bellman_error_margin,
        seed=19299
    ).plan_on(wg)
    lrtdp_res_admissible_flat = LRTDP(
        heuristic=lambda s: 50,
        bellman_error_margin=bellman_error_margin,
        seed=19299
    ).plan_on(wg)
    lrtdp_res_not_admissible = LRTDP(
        heuristic=lambda s: 0,
        bellman_error_margin=bellman_error_margin,
        seed=19299
    ).plan_on(wg)

    _test_expected_error_bound(
        bellman_error_margin=bellman_error_margin,
        optimal_res=vi_res,
        test_res=lrtdp_res_admissible_shifted,
        mdp=wg
    )
    _test_expected_error_bound(
        bellman_error_margin=bellman_error_margin,
        optimal_res=vi_res,
        test_res=lrtdp_res_admissible_flat,
        mdp=wg
    )

    with pytest.raises(OutOfExpectedErrorBound):
        # this should fail since we did not use an admissible heuristic
        _test_expected_error_bound(
            bellman_error_margin=bellman_error_margin,
            optimal_res=vi_res,
            test_res=lrtdp_res_not_admissible,
            mdp=wg
        )

def test_gridworld():
    mdp = GridWorld(
        tile_array=[
            '......g',
            '...####',
            '.###...',
            '.....##',
            '..####.',
            '..s....',
        ],
        feature_rewards={'g': 0},
        step_cost=-1,
        discount_rate=1.0
    )

    goal = mdp.absorbing_states[0]
    def heuristic(s):
        if mdp.is_absorbing(s):
            return 0.0
        return -(abs(s['x']-goal['x']) + abs(s['y']-goal['y']))

    _assert_equal_value_iteration(
        LRTDP(heuristic=lambda s: 0),
        mdp
    )
    _assert_equal_value_iteration(LRTDP(heuristic=heuristic), mdp)

def test_GNTFig6_6():
    mdp = GNTFig6_6()
    m = LRTDP(heuristic=lambda s: 0, seed=12388)
    _assert_equal_value_iteration(m, mdp)

def _assert_equal_value_iteration(planner, mdp: MarkovDecisionProcess):
    lrtdp_res = planner.plan_on(mdp)

    vi = ValueIteration()
    vi_res = vi.plan_on(mdp)

    # Ensure our VI Q values are a lower bound to the LRTDP ones.
    for s in lrtdp_res.Q.keys():
        for a in mdp.actions(s):
            assert round(vi_res.action_value[s,a], 10) <= round(lrtdp_res.Q[s][a], 10)

    def policy(s):
        return deterministic(lrtdp_res.policy.action_dist(s))

    s = deterministic(mdp.initial_state_dist())
    reachable = [s]
    MAX_ITERATIONS = 1000
    i = 0
    while reachable:
        i += 1
        if i > MAX_ITERATIONS:
            assert False, f"Unable to compare policies after {MAX_ITERATIONS} iterations"
        s = reachable.pop()
        for ns, p in mdp.next_state_dist(s, policy(s)).items():
            if p == 0:
                continue
            if not mdp.is_absorbing(ns):
                reachable.append(ns)

        # For reachable states under our policy, ensure:
        # Value is the same
        assert lrtdp_res.V[s] == vi_res.state_value[s]
        # Policy is the same, or at least our policy is something VI assigns uniform chance to
        vi_actions = ensure_uniform(vi_res.policy.action_dist(s))
        assert policy(s) in vi_actions

def test_seed_reproducibility():
    class TrialRecorder(LRTDPEventListener):
        def __init__(self):
            self.trial_data = [{
                "trial": [],
                "solved": []
            }]
        def end_of_lrtdp_timestep(self, localvars):
            pass
        def end_of_lrtdp_trial(self, localvars):
            self.trial_data.append({
                "trial": copy.deepcopy(localvars['visited']),
                "solved": copy.deepcopy(localvars['self'].res.solved),
            })

    mdp = GNTFig6_6()
    m = LRTDP(
        heuristic=lambda s: 0,
        randomize_action_order=True,
        event_listener_class=TrialRecorder,
        seed=12345
    )
    res1 = m.plan_on(mdp)

    m = LRTDP(
        heuristic=lambda s: 0,
        randomize_action_order=True,
        event_listener_class=TrialRecorder,
        seed=12345
    )
    res2 = m.plan_on(mdp)

    trials1 = res1.event_listener.trial_data
    trials2 = res2.event_listener.trial_data
    for t1, t2 in zip(trials1, trials2):
        trial1 = t1['trial']
        trial2 = t2['trial']
        for s1, s2 in zip(trial1, trial2):
            assert s1 == s2
            assert s1 in mdp.state_list

    m = LRTDP(
        heuristic=lambda s: 0,
        randomize_action_order=True,
        event_listener_class=TrialRecorder,
        seed=13004
    )
    res3 = m.plan_on(mdp)

    notequal = []
    trials3 = res3.event_listener.trial_data
    for t2, t3 in zip(trials2, trials3):
        trial2 = t2['trial']
        trial3 = t3['trial']
        for s2, s3 in zip(trial2, trial3):
            notequal.append(s3 != s2)
            assert s3 in mdp.state_list
            assert s2 in mdp.state_list
    assert any(notequal)

def test_trivial_solution():
    algo = LRTDP(
        heuristic=lambda s: 0,
        seed=42
    )
    # Normal
    mdp = DeterministicCounter(3, initial_state=0)
    R = algo.plan_on(mdp)
    assert R.V[mdp.initial_state()] == -3
    assert R.policy.run_on(mdp).action_traj == (+1, +1, +1)

    # No-op task. Now we start at 3, so value should be 0 there
    mdp = DeterministicCounter(3, initial_state=3)
    R = algo.plan_on(mdp)
    assert R.V[mdp.initial_state()] == 0
    assert R.policy.run_on(mdp).action_traj == ()
