import unittest
import numpy as np

from msdm.algorithms.laostar import LAOStar
from msdm.domains import GridWorld
from msdm.tests.domains import Counter


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
            discount_rate=1.0
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
        assert traj.state_traj[-1] == goal
        
    def test_trivial_solution(self):
        algo = LAOStar(seed=42)
        # Normal
        mdp = Counter(3, initial_state=0)
        R = algo.plan_on(mdp)
        assert R.sGraph[mdp.initial_state()]['value'] == -3
        assert R.policy.run_on(mdp).action_traj == (+1, +1, +1)

        # No-op task. Now we start at 3, so value should be 0 there
        mdp = Counter(3, initial_state=3)
        R = algo.plan_on(mdp)
        assert R.sGraph[mdp.initial_state()]['value'] == 0
        assert R.policy.run_on(mdp).action_traj == ()

if __name__ == '__main__':
    unittest.main()
