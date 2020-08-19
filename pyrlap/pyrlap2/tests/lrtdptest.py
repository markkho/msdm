import unittest

from pyrlap.pyrlap2.algorithms.lrtdp import LRTDP
from pyrlap.pyrlap2.algorithms import VectorizedValueIteration
from pyrlap.pyrlap2.tests.domains import GNTFig6_6

def deterministic(dist):
    '''
    Assumes the supplied distribution is deterministic and returns the deterministic value.
    '''
    for s in dist.support:
        if dist.prob(s) == 0:
            continue
        assert dist.prob(s) == 1
        return s

class LAOStarTestCase(unittest.TestCase):
    def test_GNTFig6_6(self):
        mdp = GNTFig6_6()
        m = LRTDP()
        m.planOn(mdp)

        vi = VectorizedValueIteration()
        vi.planOn(mdp)

        # Ensure our VI Q values are a lower bound to the LRTDP ones.
        for s in mdp.states:
            for a in mdp.actions:
                assert vi.Q[s][a] <= m.Q(mdp, s, a)

        s = deterministic(mdp.getInitialStateDist())
        reachable = [s]
        while reachable:
            s = reachable.pop()
            for ns in mdp.getNextStateDist(s, m.policy(mdp, s)).support:
                if not mdp.isTerminal(ns):
                    reachable.append(ns)

            # For reachable states under our policy, ensure:
            # Value is the same
            assert m.V[s] == vi.V[s]
            # Policy is the same
            assert m.policy(mdp, s) == deterministic(vi.policy.getActionDist(s))
        
        
if __name__ == '__main__':
    unittest.main()