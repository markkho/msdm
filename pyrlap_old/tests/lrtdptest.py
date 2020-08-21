import unittest

from pyrlap_old.pyrlap2.algorithms import VectorizedValueIteration, LRTDP
from pyrlap_old.pyrlap2.tests.domains import GNTFig6_6

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
        lrtdp_res = m.planOn(mdp)

        vi = VectorizedValueIteration()
        vi_res = vi.planOn(mdp)

        # Ensure our VI Q values are a lower bound to the LRTDP ones.
        for s in mdp.states:
            for a in mdp.actions:
                assert vi_res.Q[s][a] <= lrtdp_res.Q[s][a]

        s = deterministic(mdp.getInitialStateDist())
        reachable = [s]
        while reachable:
            s = reachable.pop()
            for ns in mdp.getNextStateDist(s, lrtdp_res.policy[s]).support:
                if not mdp.isTerminal(ns):
                    reachable.append(ns)

            # For reachable states under our policy, ensure:
            # Value is the same
            assert lrtdp_res.V[s] == vi_res.V[s]
            # Policy is the same
            assert lrtdp_res.policy[s] == deterministic(vi_res.policy.getActionDist(s))
        
        
if __name__ == '__main__':
    unittest.main()