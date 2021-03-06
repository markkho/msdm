import unittest
import numpy as np

from msdm.algorithms.laostar import LAOStar
from msdm.domains import GridWorld


np.seterr(divide='ignore')

class LAOStarTestCase(unittest.TestCase):
    def test_DeterministicLAOStarGridWorld(self):
        gw = GridWorld(
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
            termination_prob=.0
        )
        mdp = gw

        goal = mdp.absorbing_states[0]
        def heuristic(s):
            if mdp.is_terminal(s):
                return 0.0
            return -np.sum(np.abs(np.array(s['x']) - np.array(goal['x'])))

        lao = LAOStar(
            heuristic,
            max_lao_iters=100,
            policy_evaluation_iters=40,
            seed=6066253173235511770
        )
        R = lao.plan_on(mdp)
        traj = R.policy.run_on(mdp)
        print(traj.state_traj)
        assert traj.state_traj[-1] == goal
        
if __name__ == '__main__':
    unittest.main()
