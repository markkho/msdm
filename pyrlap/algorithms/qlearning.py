from pyrlap.core.util import calc_esoftmax_dist
from pyrlap.core.agent import Learner

class Qlearning(Learner):
    def __init__(self, mdp,
                 randchoose=.2,
                 softmax_temp=1,
                 discount_rate=.99,
                 learning_rate=.1,
                 eligibility_trace_method='replacement',
                 eligibility_trace_decay=0,
                 initial_qvalue=0,

                 init_state=None
                 ):
        Learner.__init__(self, mdp)

        self.randchoose = randchoose
        self.softmax_temp = softmax_temp
        self.learning_rate = learning_rate
        self.eligibility_trace_method = eligibility_trace_method
        self.eligiblity_trace_decay = eligibility_trace_decay
        self.initial_qvalue = initial_qvalue

        self.qvalues = {}
        self.eligibility_traces = {}

        if init_state is None:
            init_state = mdp.get_init_state()
        self.init_state = init_state
        if discount_rate is None:
            discount_rate = mdp.discount_rate
        self.discount_rate = discount_rate

    def _add_state_to_qvalue(self, s):
        self.qvalues[s] = {}
        for a in self.mdp.available_actions(s):
            if self.mdp.is_terminal_action(a):
                self.qvalues[s][a] = 0
            else:
                self.qvalues[s][a] = self.initial_qvalue

    def act_dist(self, s, softmax_temp=None, randchoose=None):
        # initialize qvalues if not in dictionary
        if s not in self.qvalues:
            self._add_state_to_qvalue(s)

        if softmax_temp is None:
            softmax_temp = self.softmax_temp
        if randchoose is None:
            randchoose = self.randchoose

        adist = calc_esoftmax_dist(self.qvalues[s],
                                   temp=softmax_temp, randchoose=randchoose)
        return adist

    def process(self, s, a, ns, r):
        #initialize dictionaries as needed
        if ns not in self.qvalues:
            self._add_state_to_qvalue(ns)
        if s not in self.qvalues:
            self._add_state_to_qvalue(s)

        #calculate prediction error
        max_ns_q = max(self.qvalues[ns].values())
        pred_error = r + self.discount_rate*max_ns_q - self.qvalues[s][a]

        #update eligibility traces
        # since we're off policy, if a is not the best action, then eligibility
        # traces become invalid
        a_is_max = self.qvalues[s][a] == max(self.qvalues[s].values())
        if a_is_max:
            for s_, a_et in self.eligibility_traces.items():
                for a_, e_trace in a_et.items():
                    et_update = self.discount_rate*self.eligiblity_trace_decay
                    self.eligibility_traces[s_][a_] *= et_update
        else:
            self.eligibility_traces = {}

        if self.eligiblity_trace_decay > 0:
            if s not in self.eligibility_traces:
                self.eligibility_traces[s] = {}
            if a not in self.eligibility_traces[s]:
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
            for s_, a_et in self.eligibility_traces.items():
                for a_, e_trace in a_et.items():
                    q_update = self.learning_rate*pred_error*e_trace
                    self.qvalues[s_][a_] += q_update

    def episode_reset(self):
        self.eligibility_traces = {}

    def change_learning_rate(self, new):
        self.learning_rate = new