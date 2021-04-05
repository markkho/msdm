import unittest

from msdm.algorithms import ValueIteration, LRTDP
from msdm.tests.domains import GNTFig6_6, Counter
from msdm.domains import GridWorld

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

class LRTDPTestCase(unittest.TestCase):
    def test_gridworld(self):
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
            if mdp.is_terminal(s):
                return 0.0
            return -(abs(s['x']-goal['x']) + abs(s['y']-goal['y']))

        self.assert_equal_value_iteration(LRTDP(), mdp)
        self.assert_equal_value_iteration(LRTDP(heuristic=heuristic), mdp)

    def test_GNTFig6_6(self):
        mdp = GNTFig6_6()
        m = LRTDP(seed=12388)
        self.assert_equal_value_iteration(m, mdp)

    def assert_equal_value_iteration(self, planner, mdp):
        lrtdp_res = planner.plan_on(mdp)

        vi = ValueIteration()
        vi_res = vi.plan_on(mdp)

        # Ensure our VI Q values are a lower bound to the LRTDP ones.
        for s in lrtdp_res.Q.keys():
            for a in mdp.actions(s):
                assert vi_res.Q[s][a] <= lrtdp_res.Q[s][a]

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
                if not mdp.is_terminal(ns):
                    reachable.append(ns)

            # For reachable states under our policy, ensure:
            # Value is the same
            assert lrtdp_res.V[s] == vi_res.V[s]
            # Policy is the same, or at least our policy is something VI assigns uniform chance to
            vi_actions = ensure_uniform(vi_res.policy.action_dist(s))
            assert policy(s) in vi_actions

    def test_seed_reproducibility(self):
        mdp = GNTFig6_6()
        m = LRTDP(
            randomize_action_order=True,
            seed=12345
        )
        res1 = m.plan_on(mdp)

        m = LRTDP(
            randomize_action_order=True,
            seed=12345
        )
        res2 = m.plan_on(mdp)

        for t1, t2 in zip(res1.trials, res2.trials):
            for s1, s2 in zip(t1, t2):
                assert s1 == s2

        m = LRTDP(
            randomize_action_order=True,
            seed=13
        )
        res3 = m.plan_on(mdp)

        notequal = []
        for t1, t2 in zip(res3.trials, res2.trials):
            for s1, s2 in zip(t1, t2):
                notequal.append(s1 != s2)
        assert any(notequal)

    def test_trivial_solution(self):
        algo = LRTDP(seed=42)
        # Normal
        mdp = Counter(3, initial_state=0)
        R = algo.plan_on(mdp)
        assert R.V[mdp.initial_state()] == -3
        assert R.policy.run_on(mdp).action_traj == (+1, +1, +1)

        # No-op task. Now we start at 3, so value should be 0 there
        mdp = Counter(3, initial_state=3)
        R = algo.plan_on(mdp)
        assert R.V[mdp.initial_state()] == 0
        assert R.policy.run_on(mdp).action_traj == ()


if __name__ == '__main__':
    unittest.main()
