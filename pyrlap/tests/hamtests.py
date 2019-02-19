import unittest
import numpy as np

from pyrlap.domains.taxicab import TaxiCabMDP
from pyrlap.hierarchicalrl.ham.domains.taxicab import \
    getput_hierarchy, restricted_getput, simple_getput, simple_insideoutside
from pyrlap.hierarchicalrl.ham.hierarchicalqlearning import HierarchicalQLearner

class HAMTests(unittest.TestCase):
    def setUp(self):
        self.taxicab = TaxiCabMDP()

    def test_initstate(self):
        init_state = getput_hierarchy.get_init_state('root', ())
        self.assertTrue(hasattr(init_state, 'stack'))

    def test_available_actions(self):
        init_state = getput_hierarchy.get_init_state('root', ())
        choices = getput_hierarchy.available_actions(init_state)
        self.assertTrue(isinstance(choices, list))

    def test_transition(self):
        s = getput_hierarchy.get_init_state('root', ())
        traj = []
        for _ in range(5):
            choices = getput_hierarchy.available_actions(s)
            a = choices[np.random.randint(len(choices))]
            ns, ts, r = getput_hierarchy.transition_timestep_reward(s, a)
            traj.append((s, a, ns, ts, r))
            s = ns
        self.assertTrue(len(traj) == 5)

    def test_alipqlearner_act(self):
        learner = HierarchicalQLearner(getput_hierarchy)
        s = getput_hierarchy.get_init_state('root', ())
        traj = []
        for _ in range(100):
            a = learner.act(s)
            ns, ts, r = getput_hierarchy.transition_timestep_reward(s, a)
            traj.append((s, a))
            s = ns
        self.assertTrue(len(traj) == 100)

    def test_alispqlearner_process_childcalls(self):
        learner = HierarchicalQLearner(getput_hierarchy)
        s = getput_hierarchy.get_init_state('root', ())
        a_seq = [
            ('get', (('passenger_i', 1),)),
            ('navigate', (('dest', (2, 5)),))
        ]
        a = a_seq[0]
        ns, ts, r = getput_hierarchy.transition_timestep_reward(s, a)
        learner.process(s, a, ns, ts, r)
        s = ns

        a = a_seq[1]
        ns, ts, r = getput_hierarchy.transition_timestep_reward(s, a)
        learner.process(s, a, ns, ts, r)

    def test_hierarchicalqlearner_learned_values(self):
        learner = HierarchicalQLearner(getput_hierarchy,
                                       learning_rate=1.0,
                                       discount_rate=.95,
                                       initial_qvalue=0)
        a_seq = [
            ('get', (('passenger_i', 0),)), 
            ('navigate', (('dest', (0, 0)),)), 
            ('v', ()), 
            ('v', ()), 
            ('v', ()), 
            ('pickup', ()), 
            ('put', ()),
            ('navigate', (('dest', (4, 0)),)),
            ('^', ()),
            ('^', ()),
            ('^', ()),
            ('>', ()),
            ('v', ()),
            ('v', ()),
            ('v', ()),
            ('>', ()),
            ('>', ()),
            ('>', ()),
            ('dropoff', ())
        ]
        for _ in range(50):
            s = getput_hierarchy.get_init_state('root', ())
            traj = []
            for a in a_seq:
                ns, ts, r = getput_hierarchy.transition_timestep_reward(s, a)
                if a[0] == 'pickup':
                    r += 50
                learner.process(s, a, ns, ts, r)
                traj.append((s, a, ns, ts, r))

                for a_ in getput_hierarchy.available_actions(s):
                    if a_ == a:
                        continue
                    learner._update_val(learner._comp_qvals, s, a_, -1000)
                    learner._update_val(learner._ex_qvals, s, a_, -1000)
                s = ns
            learner.episode_reset()

        dr = learner.discount_rate
        for ia, a in enumerate(a_seq):
            val = [r for s, a, ns, ts, r in traj[ia:] if ts > 0]
            val = [r*dr**ir for ir, r in enumerate(val)]
            val = sum(val)

            #calculate action value
            s, a, ns, ts, r = traj[ia]
            if ts == 1:
                action_q = r
            else:
                astack = tuple(list(s.stack) + [a,])
                action_q = []
                for s, a, ns, ts, r in traj[ia+1:]:
                    if ts == 0:
                        continue
                    if s.stack[:len(astack)] == astack:
                        action_q.append(r)
                    else:
                        break
                action_q = [r*dr**ir for ir, r in enumerate(action_q)]
                action_q = sum(action_q)

            #calculate completion value and exit value
            s, a, ns, ts, r = traj[ia]
            cur_call = s.stack
            astack = tuple(list(s.stack) + [a, ])
            completion_q = []
            exit_q = []
            if ts == 0:
                comp_steps = 0
            else:
                comp_steps = 1
            left_call = False
            for s, a, ns, ts, r in traj[ia+1:]:
                if ts == 0:
                    continue
                if s.stack[:len(astack)] == astack and not left_call:
                    comp_steps += 1
                    continue
                else:
                    left_call = True
                    if s.stack[:len(cur_call)] == cur_call:
                        completion_q.append(r)
                    else:
                        exit_q.append(r)
            exit_steps = len(completion_q) + comp_steps
            exit_q = [r * dr ** ir for ir, r in enumerate(exit_q)]
            exit_q = sum(exit_q) * dr ** exit_steps

            completion_q = [r * dr ** ir for ir, r in enumerate(completion_q)]
            completion_q = sum(completion_q) * dr ** comp_steps

            # compare learned values with true values
            s, a, ns, ts, r = traj[ia]
            learned_aval = learner._action_q(s, a)
            learned_cval = learner._completion_q(s, a)
            learned_eval = learner._external_q(s, a)
            learned_val = learner._qval(s, a)

            epsilon = 1.0e-10
            self.assertTrue(abs(exit_q - learned_eval) < epsilon)
            self.assertTrue(abs(completion_q - learned_cval) < epsilon)
            self.assertTrue(abs(action_q - learned_aval) < epsilon)
            self.assertTrue(abs(val - learned_val) < epsilon)

    def test_qlearner_learning_fullgetput(self):
        learner = HierarchicalQLearner(getput_hierarchy,
                                       learning_rate=.95,
                                       discount_rate=.99,
                                       initial_qvalue=0)
        traj = []
        for run in range(10):
            s = getput_hierarchy.get_init_state('root', ())
            for c in range(10):
                a = learner.act(s, softmax_temp=.5, randchoose=.1)
                ns, ts, r = getput_hierarchy.transition_timestep_reward(s, a)
                learner.process(s, a, ns, ts, r)
                step = dict(
                    zip(('s', 'a', 'ns', 'ts', 'r'), (s, a, ns, ts, r)))
                step['run'] = run
                step['c'] = c
                traj.append(step)
                s = ns
                if getput_hierarchy.is_terminal(s):
                    break
            learner.episode_reset()

    def test_qlearner_restgetput(self):
        learner = HierarchicalQLearner(restricted_getput,
                                       learning_rate=.95,
                                       discount_rate=.99,
                                       initial_qvalue=0)
        np.random.seed(0)
        traj = []
        for run in range(10):
            s = restricted_getput.get_init_state('root', ())
            for c in range(100):
                a = learner.act(s, softmax_temp=.5, randchoose=.1)
                ns, ts, r = restricted_getput.transition_timestep_reward(s, a)
                learner.process(s, a, ns, ts, r)
                step = dict(
                    zip(('s', 'a', 'ns', 'ts', 'r'), (s, a, ns, ts, r)))
                step['run'] = run
                step['c'] = c
                traj.append(step)
                s = ns

                if restricted_getput.is_terminal(s):
                    break
            learner.episode_reset()

    def test_qlearner_simplegetput(self):
        np.random.seed(0)
        learner = HierarchicalQLearner(simple_getput,
                                       learning_rate=.9,
                                       discount_rate=.99,
                                       initial_qvalue=0,
                                       use_state_abstraction=True)
        learner.train(episodes=200, max_choice_steps=100,
                      return_run_data=False)
        init_s = learner.ham.get_init_state()
        self.assertTrue(learner._maxqa(init_s)[0] < 180)
        traj = learner.run()
        atraj = [a for s, a, ns, r in traj]
        rtraj = [r for s, a, ns, r in traj]
        self.assertTrue(len(atraj) == 11)
        self.assertTrue(sum(rtraj) == 189)

    def test_pseudo_rewards(self):
        np.random.seed(0)
        learner = HierarchicalQLearner(simple_insideoutside,
                                       learning_rate=.9,
                                       discount_rate=.9,
                                       initial_qvalue=0,
                                       use_pseudo_rewards=True)
        learner.train(
            episodes=500, max_choice_steps=50,
            randchoose=.1, softmax_temp=2,
            return_run_data=False)
        traj = learner.run()
        atraj = [a for s, a, ns, r in traj]
        target_traj = ['>', '>', '^', '^', '^', '^',
                       '^', '<', '<', '<', '<', '<', '<']
        self.assertEqual(atraj, target_traj)

if __name__ == '__main__':
    unittest.main()
