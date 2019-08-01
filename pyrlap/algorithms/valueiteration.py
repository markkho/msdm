import logging
import warnings
import time

from pyrlap.core.agent import Planner
from pyrlap.core.util import argmax_dict, calc_esoftmax_dist, \
    calc_esoftmax_policy

logger = logging.getLogger(__name__)

class ValueIteration(Planner):
    def __init__(self, mdp,
                 transition_function=None,
                 discount_rate=.99,
                 converge_delta=.001,
                 max_iterations=100,
                 softmax_temp=0.0,
                 randchoose=0.0,
                 init_val=0.0):
        Planner.__init__(self, mdp)
        self.discount_rate = discount_rate
        self.converge_delta = converge_delta
        self.max_iterations = max_iterations
        self.softmax_temp = softmax_temp
        self.randchoose = randchoose
        self.init_val = init_val
        self.tf = transition_function

    def build_model(self):
        logger.debug('Building transition and reward model')
        start = time.time()
        tf, rf = self.mdp.get_reachable_transition_reward_functions()
        self.tf = tf
        # self.rf = rf
        logger.debug('Model built: %.2fs' % (time.time() - start))

    def solve(self):
        if self.tf is None:
            self.build_model()
        vf = {s : self.init_val for s in self.tf}
        optimal_pol = {}
        action_vals = {}
        logger.debug('Running Value Iteration')

        for i in range(self.max_iterations):
            change = 0
            vf_temp = {}
            for s, a_ns_p in self.tf.items():
                optimal_pol[s] = optimal_pol.get(s, {})
                action_vals[s] = action_vals.get(s, {})
                for a, ns_p in a_ns_p.items():

                    # calculate expected value of each action
                    action_vals[s][a] = 0
                    for ns, p in ns_p.items():
                        r = self.mdp.reward(s, a, ns)
                        ev = p*(r + self.discount_rate*vf[ns])
                        action_vals[s][a] += ev
                max_actions = argmax_dict(action_vals[s], return_one=False)
                max_actions.sort()
                optimal_pol[s] = max_actions[0]
                vf_temp[s] = action_vals[s][optimal_pol[s]]
                change = max(change, abs(vf_temp[s] - vf[s]))
            vf = vf_temp
            logger.debug('iteration: %d   change: %.2f' % (i, change))
            if change < self.converge_delta:
                break
        if change >= self.converge_delta:
            warnings.warn(
                "VI did not converge after %d iterations (delta=%.2f)" \
                % (i, change))
        self.optimal_policy = optimal_pol
        self.value_function = vf
        self.action_value_function = action_vals
        self.iterations_run = i

    def act_dist(self, s, softmax_temp=None, randchoose=None):
        if softmax_temp is None:
            softmax_temp = self.softmax_temp
        if randchoose is None:
            randchoose = self.randchoose

        return calc_esoftmax_dist(self.action_value_function[s],
                                  temp=softmax_temp,
                                  randchoose=randchoose)

    def to_dict(self, softmax_temp=None, randchoose=None):
        if softmax_temp is None:
            softmax_temp = self.softmax_temp
        if randchoose is None:
            randchoose = self.randchoose

        return calc_esoftmax_policy(self.action_value_function,
                                    temp=softmax_temp,
                                    randchoose=randchoose)


