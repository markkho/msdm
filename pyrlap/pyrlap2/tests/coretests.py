import unittest
import numpy as np
from pyrlap.pyrlap2.domains import GridWorld

class CoreTestCase(unittest.TestCase):
    def test_tabularMDP_composition(self):
        gw1 = GridWorld(
            tileArray=[
                '...g',
                '....',
                '.###',
                '...s'
            ],
            stepCost=-1,
            successProb=.99,
            terminationProb=1e-5
        )

        gw2 = GridWorld(
            tileArray=[
                '##..',
                '...g',
                '....',
                's...'
            ],
            stepCost=0,
            successProb=.99,
            terminationProb=1e-5
        )
        gw3 = gw1 & gw2

        #test state, aciton, variable composition is consistent
        self.assertTrue(tuple(gw3.states) == tuple(gw2.states))
        self.assertTrue(tuple(gw3.actions) == tuple(gw2.actions))
        self.assertTrue(tuple(gw3.variables) == tuple(gw2.variables))

        #test that mdp distributions are consistent
        s0 = gw1.initialstatevec * gw2.initialstatevec

        #test reward composition
        eqRF = (gw3.rewardmatrix == (gw1.rewardmatrix + gw2.rewardmatrix)).all()
        self.assertTrue(eqRF)

        #test simple transition composition
        tf = (gw1.transitionmatrix * gw2.transitionmatrix)
        tf = tf / tf.sum(axis=-1, keepdims=True)
        eqTF = np.isclose(gw3.transitionmatrix, tf).all()
        self.assertTrue(eqTF)

    def test_runningAgentOnMDP(self):

if __name__ == '__main__':
    unittest.main()
