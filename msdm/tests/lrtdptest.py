import unittest

from msdm.algorithms import VectorizedValueIteration, LRTDP
from msdm.tests.domains import GNTFig6_6

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
    def test_GNTFig6_6(self):
        mdp = GNTFig6_6()
        m = LRTDP()
        lrtdp_res = m.plan_on(mdp)

        vi = VectorizedValueIteration()
        vi_res = vi.plan_on(mdp)

        # Ensure our VI Q values are a lower bound to the LRTDP ones.
        for s in mdp.state_list:
            for a in mdp.action_list:
                assert vi_res.Q[s][a] <= lrtdp_res.Q[s][a]

        s = deterministic(mdp.initial_state_dist())
        reachable = [s]
        while reachable:
            s = reachable.pop()
            for ns in mdp.next_state_dist(s, lrtdp_res.policy[s]).support:
                if not mdp.is_terminal(ns):
                    reachable.append(ns)

            # For reachable states under our policy, ensure:
            # Value is the same
            assert lrtdp_res.V[s] == vi_res.V[s]
            # Policy is the same
            assert lrtdp_res.policy[s] == deterministic(vi_res.policy.action_dist(s))

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

if __name__ == '__main__':
    unittest.main()