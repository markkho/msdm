import unittest
import numpy as np

from easymdp3.domains.taxicab import TaxiCabMDP, getput_hierarchy, \
    restricted_getput, simple_getput
from easymdp3.algorithms.hierarchicalqlearning import HierarchicalQLearner
from easymdp3.core.hierarchicalrl import HAMState

class HAMTests(unittest.TestCase):
    def setUp(self):
        self.taxicab = TaxiCabMDP()
        self.full_getput = getput_hierarchy
        self.rest_getput = restricted_getput
        self.simp_getput = simple_getput

    def test_initstate(self):
        init_state = self.full_getput.get_init_state('root', ())
        self.assertTrue(hasattr(init_state, 'stack'))

    def test_available_actions(self):
        init_state = self.full_getput.get_init_state('root', ())
        choices = self.full_getput.available_actions(init_state)
        self.assertTrue(isinstance(choices, list))

    def test_transition(self):
        s = self.full_getput.get_init_state('root', ())
        traj = []
        for _ in range(5):
            choices = self.full_getput.available_actions(s)
            choice = choices[np.random.randint(len(choices))]
            ns, ts, r = self.full_getput.transition_timestep_reward(
                state=s, action=choice)
            traj.append((s, choice, ns, ts, r))
            s = ns
        self.assertTrue(len(traj) == 5)

    def test_abstract_state(self):
        s = self.full_getput.get_init_state('root', ())
        traj = []
        for t in range(20):
            choices = self.full_getput.available_actions(s)
            choice = choices[np.random.randint(len(choices))]
            ns, ts, r = self.full_getput.transition_timestep_reward(
                state=s, action=choice)
            astate = self.full_getput.get_abstract_state(s)
            traj.append((s, astate, choice, ns, ts, r))
            s = ns
        self.assertTrue(len(traj) == 20)

    def test_alipqlearner_act(self):
        learner = HierarchicalQLearner(self.full_getput)
        s = self.full_getput.get_init_state('root', ())
        traj = []
        for _ in range(100):
            a = learner.act(s)
            ns, ts, r = self.full_getput.transition_timestep_reward(
                s, a, learner.discount_rate)
            abs_s = self.full_getput.get_abstract_state(s)
            traj.append((abs_s, a))
            s = ns
        self.assertTrue(len(traj) == 100)

    def test_alispqlearner_process_childcalls(self):
        learner = HierarchicalQLearner(self.full_getput)
        s = self.full_getput.get_init_state('root', ())
        a_seq = [
            ('get', (('passenger_i', 1),)),
            ('navigate', (('dest', (2, 5)),))
        ]
        a = a_seq[0]
        ns, ts, r = self.full_getput.transition_timestep_reward(
            s, a, learner.discount_rate)
        learner.process(s, a, ns, ts, r)
        s = ns

        a = a_seq[1]
        ns, ts, r = self.full_getput.transition_timestep_reward(
            s, a, learner.discount_rate)
        learner.process(s, a, ns, ts, r)

    def test_hierarchicalqlearner_learned_values(self):
        learner = HierarchicalQLearner(self.full_getput,
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
            s = self.full_getput.get_init_state('root', ())
            traj = []
            for a in a_seq:
                ns, ts, r = self.full_getput.transition_timestep_reward(
                    s, a, learner.discount_rate)
                if a[0] == 'pickup':
                    r += 50
                learner.process(s, a, ns, ts, r)
                traj.append((s, a, ns, ts, r))

                for a_ in self.full_getput.available_actions(s):
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
        learner = HierarchicalQLearner(self.full_getput,
                                       learning_rate=.95,
                                       discount_rate=.99,
                                       initial_qvalue=0)
        traj = []
        for run in range(10):
            s = self.full_getput.get_init_state('root', ())
            for c in range(10):
                a = learner.act(s, softmax_temp=.5, randchoose=.1)
                ns, ts, r = self.full_getput.transition_timestep_reward(
                    s, a, learner.discount_rate)
                learner.process(s, a, ns, ts, r)
                step = dict(
                    zip(('s', 'a', 'ns', 'ts', 'r'), (s, a, ns, ts, r)))
                step['run'] = run
                step['c'] = c
                traj.append(step)
                s = ns
                if self.full_getput.is_terminal(s):
                    break
            learner.episode_reset()

    def test_qlearner_restgetput(self):
        learner = HierarchicalQLearner(self.rest_getput,
                                       learning_rate=.95,
                                       discount_rate=.99,
                                       initial_qvalue=0)
        np.random.seed(0)
        traj = []
        for run in range(10):
            s = self.rest_getput.get_init_state('root', ())
            for c in range(100):
                a = learner.act(s, softmax_temp=.5, randchoose=.1)
                ns, ts, r = self.rest_getput.transition_timestep_reward(
                    s, a, learner.discount_rate)
                learner.process(s, a, ns, ts, r)
                step = dict(
                    zip(('s', 'a', 'ns', 'ts', 'r'), (s, a, ns, ts, r)))
                step['run'] = run
                step['c'] = c
                traj.append(step)
                s = ns

                if self.rest_getput.is_terminal(s):
                    break
            learner.episode_reset()

    def test_qlearner_simplegetgput(self):
        np.random.seed(0)
        learner = HierarchicalQLearner(self.simp_getput,
                                       learning_rate=.9,
                                       discount_rate=.99,
                                       initial_qvalue=0)
        learner.train(episodes=200, max_choice_steps=100,
                      return_run_data=False)
        init_s = learner.ham.get_init_state()
        self.assertTrue(learner._maxqa(init_s)[0] < 180)
        traj = learner.run()
        atraj = [a[0] for s, a, ns, r in traj]
        rtraj = [r for s, a, ns, r in traj]
        self.assertTrue(len(atraj) == 12)
        self.assertTrue(sum(rtraj) == 188)

if __name__ == '__main__':
    unittest.main()
