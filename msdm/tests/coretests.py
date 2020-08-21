import unittest
import numpy as np
from msdm.domains import GridWorld
from msdm.algorithms import VectorizedValueIteration
from msdm.core.assignment import DefaultAssignmentMap, AssignmentMap

np.seterr(divide='ignore')

class CoreTestCase(unittest.TestCase):
    def test_tabularMDP_ANDcomposition(self):
        gw1 = GridWorld(
            tile_array=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            step_cost=-1,
            success_prob=.99,
            termination_prob=1e-5
        )

        gw2 = GridWorld(
            tile_array=[
                '##..',
                's..g',
                '....',
                's...'
            ],
            step_cost=0,
            success_prob=.99,
            termination_prob=1e-5
        )
        gw3 = gw1 & gw2

        #test state, aciton, variable composition is consistent
        self.assertTrue(all(s3 == s2 for s3, s2 in zip(gw3.state_list, gw2.state_list)))
        self.assertTrue(all(a3 == a2 for a3, a2 in zip(gw3.action_list, gw2.action_list)))

        #test that mdp distributions are consistent
        s0 = gw1.initialstatevec * gw2.initialstatevec
        s0 = s0 / s0.sum()
        eqS0 = np.isclose(gw3.initialstatevec, s0).all()
        self.assertTrue(eqS0)

        #test that less than or equal are reachable
        rv = sum(gw1.reachablestatevec * gw2.reachablestatevec)
        self.assertTrue(sum(gw3.reachablestatevec) <= rv)

        #test reward composition
        rs = gw3.reachablestatevec*gw1.reachablestatevec*gw2.reachablestatevec
        ast = gw3.absorbingstatevec*gw1.absorbingstatevec*gw2.absorbingstatevec
        ignore = rs[None, None, :]*rs[:, None, None]*ast[:, None, None]*ast[None, None, :] 
        eqRF = (ignore*gw3.rewardmatrix == ignore*(gw1.rewardmatrix + gw2.rewardmatrix)).all()
        self.assertTrue(eqRF)

        #test simple transition composition
        tf = (gw1.transitionmatrix * gw2.transitionmatrix)
        tf = tf / tf.sum(axis=-1, keepdims=True)
        eqTF = np.isclose(gw3.transitionmatrix, tf).all()
        self.assertTrue(eqTF)

    def test_runningAgentOnMDP(self):
        gw1 = GridWorld(
            tile_array=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            step_cost=-1,
        )
        vi = VectorizedValueIteration(temperature=.1,
                                      entropy_regularization=True)
        res = vi.plan_on(gw1)
        stateTraj = res.policy.run_on(gw1)['stateTraj']
        self.assertTrue(stateTraj[-2] in gw1.absorbing_states)
        self.assertTrue(gw1.is_terminal(stateTraj[-1]))
        self.assertTrue(stateTraj[0] in gw1.initial_states)

    def test_AssignmentMap_encode(self):
        m = AssignmentMap()
        keys = [
            'ñ',
            b'hi',
            [3, 4],
            (1, 2),
            {'hi': 3},
            3,
        ]
        for key in keys:
            # Testing setter
            m[key] = 1337

        # Making sure we can also list keys
        assert len(list(m.keys())) == len(keys)
        for el in m.keys():
            assert el in keys

    def test_DefaultAssignmentMap(self):
        m = DefaultAssignmentMap(lambda: 3)
        assert m['number'] == 3
        m['number'] = 7
        assert m['number'] == 7
        del m['number']
        assert m['apples'] == 3

        m = DefaultAssignmentMap(lambda key: key * 2)
        assert m[3] == 6
        m[3] = 7
        assert m[3] == 7
        del m[3]
        assert m[3] == 6

if __name__ == '__main__':
    unittest.main()
