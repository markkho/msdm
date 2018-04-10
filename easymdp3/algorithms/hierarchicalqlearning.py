import numpy as np
import logging

from easymdp3.core.util import calc_esoftmax_dist, sample_prob_dict
from easymdp3.core.hierarchicalrl import HAMState

logger = logging.getLogger(__name__)

class HierarchicalQLearner(object):
    def __init__(self, ham,
                 randchoose=.2,
                 softmax_temp=1,
                 discount_rate=.99,
                 learning_rate=.9,
                 initial_qvalue=0):
        self.ham = ham
        self.softmax_temp = softmax_temp
        self.randchoose = randchoose
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.init_q = initial_qvalue

        self._ground_q = {}
        self._comp_qvals = {}
        self._ex_qvals = {}

        # for tracking timesteps that a policy was called
        self.stack_to_timesteps = {}
        self.stack_to_last_stateaction = {}

    def _update_val(self, dictionary, s, a, update_val,
                    learning_rate=None,
                    timesteps=0):
        if learning_rate is None:
            learning_rate = self.learning_rate
        dictionary[s] = dictionary.get(s, {})
        old_val = dictionary[s].get(a, self.init_q)
        new_val = (1 - learning_rate)*old_val \
                  + learning_rate*(self.discount_rate**timesteps)*update_val
        dictionary[s][a] = new_val

    def _action_q(self, s, a):
        ttype = self.ham.transition_type(s, a)
        if ttype == 'ground':
            a_qs = self._ground_q.get(s, {})
            q = a_qs.get(a, self.init_q)
            return q
        elif ttype == 'selfloop':
            # taking action a immediately returns to calling context
            #this is bad since it leads to a costless loop - never do this!
            return -np.inf
        elif ttype == 'termination':
            return 0
        else:
            s_, _, _ = self.ham.transition_timestep_reward(s, a)
            max_q = -np.inf
            for a_ in self.ham.available_actions(s_):
                child_act_q = self._action_q(s_, a_)
                child_comp_q = self._completion_q(s_, a_)
                child_q = child_act_q + child_comp_q
                if child_q > max_q:
                    max_q = child_q
            return max_q

    def _completion_q(self, s, a):
        a_qs = self._comp_qvals.get(s, {})
        q = a_qs.get(a, self.init_q)
        return q

    def _external_q(self, s, a):
        a_qs = self._ex_qvals.get(s, {})
        q = a_qs.get(a, self.init_q)
        return q

    def _qval(self, s, a):
        act_q = self._action_q(s, a)
        comp_q = self._completion_q(s, a)
        ex_q = self._external_q(s, a)
        return act_q + comp_q + ex_q

    def _qvals(self, s):
        actions = self.ham.available_actions(s)
        qs = [self._qval(s, a) for a in actions]
        return dict(zip(actions, qs))

    def _completion_qs(self, s):
        actions = self.ham.available_actions(s)
        qs = [self._completion_q(s, a) for a in actions]
        return dict(zip(actions, qs))

    def _action_qs(self, s):
        actions = self.ham.available_actions(s)
        qs = [self._action_q(s, a) for a in actions]
        return dict(zip(actions, qs))

    def _external_qs(self, s):
        actions = self.ham.available_actions(s)
        qs = [self._external_q(s, a) for a in actions]
        return dict(zip(actions, qs))

    def _maxqa(self, s):
        actions = self.ham.available_actions(s)
        qs = [self._qval(s, a) for a in actions]
        max_q = max(qs)
        return (max_q, actions[qs.index(max_q)])


    def act(self, s, softmax_temp=None, randchoose=None):
        # abs_s = self.ham.get_abstract_state(s)
        actions = self.ham.available_actions(s)
        a_q = {}
        for a in actions:
            q = self._qval(s, a)
            if q == -np.inf: #never take actions that machine loop
                continue
            a_q[a] = q

        if softmax_temp is None:
            softmax_temp = self.softmax_temp
        if randchoose is None:
            randchoose = self.randchoose

        adist = calc_esoftmax_dist(a_q,
                                   temp=softmax_temp,
                                   randchoose = randchoose)
        return sample_prob_dict(adist)

    def _update_non_terminated_completion_exit_qs(self, update_stack, ns):
        # Get the next state's value
        max_nq, max_na = self._maxqa(ns)

        # Get the state, action, and timesteps associated with the updating
        # context/stack
        last_s, last_a = self.stack_to_last_stateaction[update_stack]
        del self.stack_to_last_stateaction[update_stack]
        last_sa_ts = self.stack_to_timesteps[update_stack]
        del self.stack_to_timesteps[update_stack]

        if last_s is None:
            return

        # Update the last state/action completion q-value using the next state
        next_act_val = self._action_q(ns, max_na)
        next_comp_val = self._completion_q(ns, max_na)
        next_actcomp_val = next_act_val + next_comp_val
        self._update_val(
            self._comp_qvals, last_s, last_a, next_actcomp_val,
            timesteps=last_sa_ts)

        # Update the next context's last state-action's exit q-value
        next_ex_val = self._external_q(ns, max_na)
        self._update_val(
            self._ex_qvals, last_s, last_a, next_ex_val,
            timesteps=last_sa_ts)

    def _update_terminated_exit_qs(self, exiting_stack, ns):
        # Get the next context's next state's value
        max_nq, max_na = self._maxqa(ns)

        # Next context's last state, last action, and timesteps since then
        last_s, last_a = self.stack_to_last_stateaction[exiting_stack]
        del self.stack_to_last_stateaction[exiting_stack]
        last_sa_ts = self.stack_to_timesteps[exiting_stack]
        del self.stack_to_timesteps[exiting_stack]

        if last_s is None:
            return

        # In MAXQ, this doesn't get updated and is fixed as a pseudoreward

        # note: this averages over all exit states in the abstracted
        # case see Andre and Russell 2002
        self._update_val(self._ex_qvals, last_s, last_a, max_nq,
            timesteps=last_sa_ts)

    def process(self, s, a, ns, ts, r):
        # update ground state action value
        if self.ham.is_ground_action(a):
            self._update_val(self._ground_q, s, a, r)

        #update timesteps
        for stack_i in range(len(s.stack)):
            substack = s.stack[:stack_i + 1]
            #initialize timestep count and record entrance sa for immediate
            # context
            if (stack_i + 1) == len(s.stack) or \
                    substack not in self.stack_to_timesteps:
                self.stack_to_timesteps[substack] = 0
                self.stack_to_last_stateaction[substack] = (s, a)
            self.stack_to_timesteps[substack] += ts

        #if the non-max action was taken, we need to reset the trace
        # since it "doesn't count"
        max_q, max_a = self._maxqa(s)
        if self._qval(s, a) < max_q:
            for substack in self.stack_to_last_stateaction:
                self.stack_to_last_stateaction[substack] = (None, None)

        # Called a ground action or returning from child processes
        if len(s.stack) >= len(ns.stack):
            # for the next subroutine context, update the completion and exit
            # q-values of its last visited state/action
            self._update_non_terminated_completion_exit_qs(ns.stack, ns)

        # Check exit conditions and update exit states
        if len(s.stack) > len(ns.stack):
            # for the contexts we exited, check if they validly terminated
            # if so, update their exit values
            for stack_i in range(len(ns.stack) + 1, len(s.stack) + 1):
                exiting_stack = s.stack[:stack_i]
                test_s = HAMState(groundstate=ns.groundstate,
                                  stack=exiting_stack)
                if self.ham.subtask_terminates(test_s):
                    self._update_terminated_exit_qs(exiting_stack, ns)
                else:
                    del self.stack_to_last_stateaction[exiting_stack]
                    del self.stack_to_timesteps[exiting_stack]

    def episode_reset(self):
        self.stack_to_timesteps = {}
        self.stack_to_last_stateaction = {}

    def train(self, episodes=100, max_choice_steps=100,
              softmax_temp=0.0, randchoose=0.05, return_run_data=True):
        run_data = []
        for run in range(episodes):
            s = self.ham.get_init_state('root', ())
            for c in range(max_choice_steps):
                a = self.act(s, softmax_temp=softmax_temp,
                             randchoose=randchoose)
                ns, ts, r = self.ham.transition_timestep_reward(
                    s, a, self.discount_rate)
                self.process(s, a, ns, ts, r)
                if return_run_data:
                    step = dict(
                        zip(('s', 'a', 'ns', 'ts', 'r'), (s, a, ns, ts, r)))
                    step['run'] = run
                    step['c'] = c
                    run_data.append(step)
                s = ns
                if self.ham.is_terminal(s):
                    break
            if run % 100 == 0:
                logger.debug('run: %d ; steps: %d' % (run, c))
            self.episode_reset()

        if return_run_data:
            return run_data

    def run(self, softmax_temp=0, randchoose=0, max_steps=1000):
        traj = []
        s = self.ham.get_init_state()
        for _ in range(max_steps):
            a = self.act(s, softmax_temp=softmax_temp, randchoose=randchoose)
            ns, ts, r = self.ham.transition_timestep_reward(s, a)
            if self.ham.is_terminal(ns):
                break
            if self.ham.is_ground_action(a):
                traj.append((s.groundstate, a, ns.groundstate, r))
            s = ns
        return traj