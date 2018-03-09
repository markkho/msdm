from __future__ import division

from ..util import calc_softmax_policy, calc_stochastic_policy

import numpy as np

class Qlearning(object):
    def __init__(self, mdp,
                 decision_rule='egreedy',
                 egreedy_epsilon=.2,
                 softmax_temp=1,
                 discount_rate=None,
                 learning_rate=.1,
                 eligibility_trace_method='replacement',
                 eligiblity_trace_decay=0,
                 initial_qvalue=0,

                 init_state=None
                 ):

        self.mdp = mdp
        self.decision_rule = decision_rule
        self.egreedy_epsilon = egreedy_epsilon
        self.softmax_temp = softmax_temp
        self.learning_rate = learning_rate
        self.eligibility_trace_method = eligibility_trace_method
        self.eligiblity_trace_decay = eligiblity_trace_decay
        self.initial_qvalue = initial_qvalue

        self.qvalues = {}
        self.eligibility_traces = {}

        if init_state is None:
            init_state = mdp.get_init_state()
        self.init_state = init_state
        if discount_rate is None:
            discount_rate = mdp.discount_rate
        self.discount_rate = discount_rate


    def get_action(self, s):
        #initialize qvalues if not in dictionary
        if s not in self.qvalues:
            self.qvalues[s] = {}
            for a in self.mdp.available_actions(s):
                self.qvalues[s][a] = self.initial_qvalue

        #special case of a single action
        if len(self.qvalues[s]) == 1:
            return self.qvalues[s].keys()[0]

        #select a decision rule
        qs = list(self.qvalues[s].iteritems())
        if self.decision_rule == 'egreedy':
            if np.random.random() > self.egreedy_epsilon:
                max_q = max(qs, key=lambda aq: aq[1])[1]
                maxactions = [a for a, q in qs if q == max_q]
                max_i = np.random.choice(range(len(maxactions)))
                return maxactions[max_i]
            else:
                ai = np.random.choice(range(len(qs)))
                return qs[ai][0]
        elif self.decision_rule == 'softmax':
            qvals = np.array([q for a, q in qs])
            qvals = np.exp(qvals/self.softmax_temp)
            probs = qvals/np.sum(qvals)
            actions = [a for a, q in qs]
            return actions[np.random.choice(range(len(qs)), p=probs)]


    def process(self, s, a, ns, r):
        #update dictionaries as needed
        if ns not in self.qvalues:
            self.qvalues[ns] = {}
            for a_ in self.mdp.available_actions(ns):
                self.qvalues[ns][a_] = self.initial_qvalue

        if s not in self.qvalues:
            self.qvalues[s] = {}
            for a_ in self.mdp.available_actions(s):
                self.qvalues[s][a_] = self.initial_qvalue

        #calculate prediction error
        max_ns_q = max(self.qvalues[ns].values())
        pred_error = r + self.discount_rate*max_ns_q - self.qvalues[s][a]

        #update eligibility traces
        if s not in self.eligibility_traces:
            self.eligibility_traces[s] = {}
            for a in self.mdp.available_actions(s):
                self.eligibility_traces[s][a] = 0
        original_trace_val = self.eligibility_traces[s][a]
        s_actions = self.mdp.available_actions(s)
        self.eligibility_traces[s] = {a_: 0 for a_ in s_actions}
        if self.eligibility_trace_method == 'replacement':
            self.eligibility_traces[s][a] = 1
        elif self.eligibility_trace_method == 'cumulative':
            self.eligibility_traces[s][a] = original_trace_val + 1

        #update q values
        if self.eligiblity_trace_decay == 0:
            q_update = self.learning_rate * pred_error
            self.qvalues[s][a] += q_update
        else:
            for s_, a_q in self.qvalues.iteritems():
                for a_ in a_q.iterkeys():
                    if s_ in self.eligibility_traces:
                        e_trace = self.eligibility_traces[s_][a_]
                    else:
                        e_trace = 0
                    q_update = self.learning_rate*pred_error*e_trace
                    self.qvalues[s_][a_] += q_update

                    e_trace_update = self.discount_rate\
                                     *self.eligiblity_trace_decay
                    self.eligibility_traces[s_][a_] *= e_trace_update

    def reset_eligibility_traces(self):
        self.eligibility_traces = {}

    def run(self,
            episodes=20, max_steps=100,
            init_state=None):
        if init_state is None:
            init_state = self.init_state


        run_data = []

        for e in xrange(episodes):
            s = init_state
            for t in xrange(max_steps):
                a = self.get_action(s)
                ns = self.mdp.transition(s=s, a=a)
                r = self.mdp.reward(s=s, a=a, ns=ns)
                run_data.append({
                    'episode': e, 'timestep': t,
                    's': s, 'a': a, 'ns': ns, 'r': r
                })
                self.process(s, a, ns, r)
                s = ns
                if self.mdp.is_terminal(ns):
                    break
        return run_data

    def get_softmax_policy(self, temp=1):
        return calc_softmax_policy(self.qvalues, temp)

    def get_egreedy_policy(self, rand_choose=.2):
        return calc_stochastic_policy(self.qvalues, rand_choose)