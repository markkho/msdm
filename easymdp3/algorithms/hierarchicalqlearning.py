import numpy as np
import logging

from easymdp3.core.util import calc_esoftmax_dist, sample_prob_dict
from easymdp3.core.hierarchicalrl import HAMState

logger = logging.getLogger(__name__)

# =========================================== #
#     State Abstraction Function Wrapper      #
# =========================================== #
def state_abstraction_wrapper(func):
    def abstract_and_call(self, s, *args, **kwargs):

        if not self.use_state_abstraction:
            return func(self, s, *args, **kwargs)

        pname, pparams = s.stack[-1]
        process = self.ham.abstract_machines[pname]
        abs_s = process.state_abstraction(s.groundstate, s.stack,
                                          **dict(pparams))
        return func(self, abs_s, *args, **kwargs)
    return abstract_and_call

class HierarchicalQLearner(object):
    def __init__(self, ham,
                 randchoose=.2,
                 softmax_temp=1,
                 discount_rate=.99,
                 learning_rate=.9,
                 initial_qvalue=0,
                 use_state_abstraction=False,
                 use_pseudo_rewards=None):
        self.ham = ham
        self.softmax_temp = softmax_temp
        self.randchoose = randchoose
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.init_q = initial_qvalue
        if use_pseudo_rewards is None:
            use_pseudo_rewards = ham.use_pseudo_rewards
        self.use_pseudo_rewards = use_pseudo_rewards
        self.use_state_abstraction = use_state_abstraction

        self._ground_q = {}
        self._comp_qvals = {}
        self._ex_qvals = {}

        # for tracking timesteps that a policy was called
        self.stack_to_timesteps = {}
        self.stack_to_last_stateaction = {}

        self._vmin = -1000000000





    # =========================================== #
    #      Methods for inspecting Q-values        #
    # =========================================== #
    @state_abstraction_wrapper
    def _action_q(self, s, a, ns_available_actions=None):
        if self.ham.is_ground_action(a):
            a_qs = self._ground_q.get(s, {})
            q = a_qs.get(a, self.init_q)
            return q

        if self.ham.is_termination_action(a):
            return 0

        ns_ts_r_dist = self.ham.transition_timestep_reward_dist(s, a)
        ns, _, _ = sample_prob_dict(ns_ts_r_dist)
        if ns_available_actions is None:
            ns_available_actions = self.ham.available_actions(ns)

        #self-loop - we generally don't want this
        if ns == s:
            return self._vmin

        max_q = -np.inf
        for a_ in ns_available_actions:
            child_act_q = self._action_q(ns, a_)
            child_comp_q = self._completion_q(ns, a_)
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

    def _q_decomposition(self, s, a=None):
        actions = self.ham.available_actions(s)
        a_qs = {}
        a_aqs = self._action_qs(s)
        a_cqs = self._completion_qs(s)
        a_eqs = self._external_qs(s)
        for a_ in actions:
            a_qs[a_] = {
                'action': a_aqs[a_],
                'completion': a_cqs[a_],
                'external': a_eqs[a_]
            }
        if a is not None:
            return a_qs[a]
        return a_qs

    # =========================================== #
    #      Methods for updating Q-values          #
    # =========================================== #
    def _update_ground_q(self, s, a, v, ts=0):
        self._update_val(self._ground_q, s, a, v, timesteps=ts)

    def _update_comp_q(self, s, a, v, ts=0):
        self._update_val(self._comp_qvals, s, a, v, timesteps=ts)

    def _update_ext_q(self, s, a, v, ts=0):
        self._update_val(self._ex_qvals, s, a, v, timesteps=ts)

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

    # =========================================== #
    #          Learning Agent Interface           #
    # =========================================== #
    def act(self, s, softmax_temp=None, randchoose=None):
        actions = self.ham.available_actions(s)
        a_q = {}
        for a in actions:
            q = self._qval(s, a)
            # if q == -np.inf: #never take actions that machine loop
            #     continue
            a_q[a] = q

        if softmax_temp is None:
            softmax_temp = self.softmax_temp
        if randchoose is None:
            randchoose = self.randchoose

        adist = calc_esoftmax_dist(a_q,
                                   temp=softmax_temp,
                                   randchoose = randchoose)
        return sample_prob_dict(adist)

    def process(self, s, a, ns, ts, r):
        # update ground state action value
        if self.ham.is_ground_action(a):
            self._update_val(self._ground_q, s, a, r)

        # update timesteps at each level of current stack
        for stack_i in range(len(s.stack)):
            substack = s.stack[:stack_i + 1]
            if (stack_i + 1) == len(s.stack):
                self.stack_to_timesteps[substack] = 0
                self.stack_to_last_stateaction[substack] = (s, a)
            self.stack_to_timesteps[substack] += ts

        # if the non-max action was taken, we need to reset the trace
        # (similar to Q(lambda))
        max_q, max_a = self._maxqa(s)
        if self._qval(s, a) < max_q:
            for substack in self.stack_to_last_stateaction:
                if self.stack_to_last_stateaction[substack] == (s, a):
                    #we don't need to reset things for the current step
                    continue
                self.stack_to_last_stateaction[substack] = (None, None)

        # ========================== #
        #  Update Q-value estimates  #
        # ========================== #
        max_nq, max_na = self._maxqa(ns)

        # Update state, actions associated with returned substacks
        # Note: this won't update anything if the stack was extended
        for stack_i in range(len(ns.stack), len(s.stack) + 1):
            update_stack = s.stack[:stack_i]

            last_s, last_a = self.stack_to_last_stateaction[update_stack]
            last_sa_ts = self.stack_to_timesteps[update_stack]
            del self.stack_to_last_stateaction[update_stack]
            del self.stack_to_timesteps[update_stack]

            if last_s is None:
                continue

            # Current calling context update
            if stack_i == len(ns.stack):
                next_act_val = self._action_q(ns, max_na)
                next_comp_val = self._completion_q(ns, max_na)
                next_actcomp_val = next_act_val + next_comp_val
                self._update_comp_q(last_s, last_a,
                                    next_actcomp_val, last_sa_ts)

                # Update the next context's last state-action's exit q-value
                next_ex_val = self._external_q(ns, max_na)
                self._update_ext_q(last_s, last_a, next_ex_val, last_sa_ts)
                continue

            # Terminated calling context update
            test_s = HAMState(groundstate=ns.groundstate,
                              stack=update_stack)
            if self.ham.subtask_validly_terminates(test_s):
                #Using pseudo-rewards implements MAXQ, whereby the exit-value
                # for a terminal state is set to a particular pseudo-reward
                # rather than the true value of the resulting state
                if self.use_pseudo_rewards:
                    pseudo_r = self.ham.get_pseudo_reward(test_s)
                    self._update_ext_q(last_s, last_a, pseudo_r, last_sa_ts)
                else:
                    # Note: this averages over all exit states in the
                    # abstracted case see Andre and Russell 2002
                    self._update_ext_q(last_s, last_a, max_nq, last_sa_ts)

    # ============================ #
    #     Training Interface       #
    # ============================ #

    def episode_reset(self):
        self.stack_to_timesteps = {}
        self.stack_to_last_stateaction = {}

    def train(self, episodes=100, max_choice_steps=100,
              softmax_temp=0.0, randchoose=0.05, return_run_data=True):
        run_data = []
        for episode in range(episodes):
            s = self.ham.get_init_state('root', ())
            for c in range(max_choice_steps):
                a = self.act(s, softmax_temp=softmax_temp,
                             randchoose=randchoose)
                ns, ts, r = self.ham.transition_timestep_reward(s, a)
                self.process(s, a, ns, ts, r)
                if return_run_data:
                    step = dict(
                        zip(('s', 'a', 'ns', 'ts', 'r'), (s, a, ns, ts, r)))
                    step['episode'] = episode
                    step['c'] = c
                    run_data.append(step)
                s = ns
                if self.ham.is_terminal(s):
                    break
            if episode % 100 == 0:
                logger.debug('run: %d ; steps: %d' % (episode, c))
            self.episode_reset()

        if return_run_data:
            return run_data

    def run(self, softmax_temp=0, randchoose=0, max_steps=1000,
            return_only_ground_traj=True):
        traj = []
        s = self.ham.get_init_state()
        for _ in range(max_steps):
            a = self.act(s, softmax_temp=softmax_temp, randchoose=randchoose)
            ns, ts, r = self.ham.transition_timestep_reward(s, a)
            if self.ham.is_terminal(ns):
                break
            if return_only_ground_traj:
                if self.ham.is_ground_action(a):
                    traj.append((s.groundstate, a[0], ns.groundstate, r))
            else:
                traj.append((s, a, ns, r))
            s = ns
        return traj