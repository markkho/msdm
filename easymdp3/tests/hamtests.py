import unittest
import numpy as np

from easymdp3.core.hierarchicalrl import HAMInterpreter
from easymdp3.domains.taxicab import TaxiCabMDP, getput_hierarchy

class HAMTests(unittest.TestCase):
    def setUp(self):
        self.taxicab = TaxiCabMDP()
        self.hamint = HAMInterpreter(self.taxicab, getput_hierarchy)

    def test_initstate(self):
        init_state = self.hamint.get_init_state('root', ())
        self.assertTrue(hasattr(init_state, 'stack'))

    def test_available_actions(self):
        init_state = self.hamint.get_init_state('root', ())
        choices = self.hamint.available_actions(**init_state._asdict())
        self.assertTrue(isinstance(choices, list))

    def test_transition(self):
        s = self.hamint.get_init_state('root', ())
        traj = []
        for _ in range(5):
            choices = self.hamint.available_actions(**s._asdict())
            choice = choices[np.random.randint(len(choices))]
            ns, ts, r = self.hamint.transition_timestep_reward(
                state=s.state, stack=s.stack, a=choice)
            traj.append((s, choice, ns, ts, r))
            s = ns
        self.assertTrue(len(traj) == 5)

    def test_abstract_state(self):
        s = self.hamint.get_init_state('root', ())
        traj = []
        # choice_list = [('get', (('passenger_i', 0),)),
        #                ('navigate', (('dest', (0, 0)),)),
        #                ('v', ()),
        #                ('v', ()),
        #                ('v', ()),
        #                ('pickup', ()),
        #                ('terminate', ())]
        for t in range(20):
            choices = self.hamint.available_actions(**s._asdict())
            choice = choices[np.random.randint(len(choices))]
            ns, ts, r = self.hamint.transition_timestep_reward(
                state=s.state, stack=s.stack, a=choice)
            astate = self.hamint.get_abstract_state(
                state=s.state, stack=s.stack)
            traj.append((s, astate, choice, ns, ts, r))
            s = ns
        self.assertTrue(len(traj) == 20)

if __name__ == '__main__':
    unittest.main()
