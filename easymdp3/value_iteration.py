from __future__ import division
import logging
import warnings

import numpy as np
from random import random, randint

import rf_utils

logger = logging.getLogger(__name__)

def VI(rf, tf, init_state=None, max_iterations=100, delta=.001,
       gamma=.99, fixed_action_order=True, print_info=False):
    """Finds an optimal deterministic policy given a
            reward function: {s: {a: {ns : r } }, and 
            transition function: {s : {a : {ns : prob}}}
    """
    if type(rf.values()[0]) != dict:
        rf = rf_utils.toStateActionNextstateRF(rf, tf)
    elif type(rf.values()[0].values()[0]) != dict:
        rf = rf_utils.toStateActionNextstateRF(rf, tf)

    states = [s for s in tf.keys()]
    if fixed_action_order:
        #always consistent ordering of actions
        state_actions = dict([(s, sorted(a.keys())) for s, a in tf.items()])
    else:
        #random but consistent ordering of actions
        state_actions = {}
        for s, a in tf.iteritems():
            state_actions[s] = sorted(a.keys(), key=lambda _ : random())
    vf = dict([(s, 0.0) for s in states])
    op = {}
    action_vals = {}
    for s, actions in state_actions.iteritems():
        op[s] = actions[randint(0, len(actions)-1)]
        action_vals[s] = dict(zip(actions, [0.0]*len(actions)))

    for i in range(max_iterations):
        change = 0
        vf_temp = {}
        for state, actions in state_actions.iteritems():
            max_action = actions[0]
            max_action_val = -np.inf
            for action in actions:
                #calculate expected utility of each action
                action_vals[state][action] = 0
                for ns, prob in tf[state][action].iteritems():
                    update = prob*(rf[state][action][ns] + gamma*vf[ns])
                    action_vals[state][action] += update
                if max_action_val < action_vals[state][action]:
                    max_action = action
                    max_action_val = action_vals[state][action]
            vf_temp[state] = max_action_val
            op[state] = max_action
            change = max(change, abs(vf_temp[state]-vf[state]))
        vf = vf_temp
        logger.debug('iteration: %d   change: %.2f' % (i, change))
        if change < delta:
            break
    if change >= delta:
        warnings.warn("VI did not converge after %d iterations (delta=%.2f)" \
                      % (i, change))
    return op, vf, action_vals