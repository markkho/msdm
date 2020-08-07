import unittest
import numpy as np
from pyrlap.pyrlap2.core import TERMINALSTATE
from pyrlap.pyrlap2.domains import GridWorld
from pyrlap.pyrlap2.algorithms import VectorizedValueIteration

np.seterr(divide='ignore')

class CoreTestCase(unittest.TestCase):
    def test_tabularMDP_ANDcomposition(self):
        gw1 = GridWorld(
            tileArray=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            stepCost=-1,
            successProb=.99,
            terminationProb=1e-5
        )

        gw2 = GridWorld(
            tileArray=[
                '##..',
                's..g',
                '....',
                's...'
            ],
            stepCost=0,
            successProb=.99,
            terminationProb=1e-5
        )
        gw3 = gw1 & gw2

        #test state, aciton, variable composition is consistent
        self.assertTrue(all(s3 == s2 for s3, s2 in zip(gw3.states, gw2.states)))
        self.assertTrue(all(a3 == a2 for a3, a2 in zip(gw3.actions, gw2.actions)))

        #test that mdp distributions are consistent
        s0 = gw1.initialstatevec * gw2.initialstatevec
        s0 = s0 / s0.sum()
        eqS0 = np.isclose(gw3.initialstatevec, s0).all()
        self.assertTrue(eqS0)

        #test that less than or equal are reachable
        rv = sum(gw1.reachablestatevec * gw2.reachablestatevec)
        self.assertTrue(sum(gw3.reachablestatevec) <= rv)

        #test reward composition
        eqRF = (gw3.rewardmatrix == (gw1.rewardmatrix + gw2.rewardmatrix)).all()
        self.assertTrue(eqRF)

        #test simple transition composition
        tf = (gw1.transitionmatrix * gw2.transitionmatrix)
        tf = tf / tf.sum(axis=-1, keepdims=True)
        eqTF = np.isclose(gw3.transitionmatrix, tf).all()
        self.assertTrue(eqTF)

    def test_runningAgentOnMDP(self):
        gw1 = GridWorld(
            tileArray=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            stepCost=-1,
        )
        vi = VectorizedValueIteration(temperature=.1,
                                      entropyRegularization=True)
        vi.planOn(gw1)
        stateTraj = vi.policy.runOn(gw1)['stateTraj']
        self.assertTrue(stateTraj[-2] in gw1.absorbingStates)
        self.assertTrue(gw1.isTerminal(stateTraj[-1]))
        self.assertTrue(stateTraj[0] in gw1.initStates)

if __name__ == '__main__':
    unittest.main()
