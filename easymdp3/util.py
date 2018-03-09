from __future__ import division
import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)
#================================#
#
#    Functions related to policies
#    and probabilities
#
#================================#
np.seterr(all='raise')

def calc_softmax_policy(stateaction_vals, temp=1):
    soft_max_policy = {}
    for s, a_q in stateaction_vals.iteritems():
        a_q = a_q.items()
        try:
            sm = np.exp([(q/temp) for a, q in a_q])
            sm = list(sm/np.sum(sm))
        except FloatingPointError:
            sm = np.empty((len(a_q),))
            sm[:] = np.nan
            warnings.warn("Error computing softmax values")
        soft_max_policy[s] = dict(zip([a for a, q in a_q], sm))
    return soft_max_policy

def calc_softmax_dist(action_vals, temp=1):
    actions, qs = zip(*action_vals.items())
    try:
        sm = np.exp([q/temp for q in qs])
        sm = sm/np.sum(sm)
    except FloatingPointError:
        sm = np.empty((len(actions),))
        sm[:] = np.nan
        warnings.warn("Error computing softmax values")
    return dict(zip(actions, sm))

def calc_stochastic_policy(action_vals, rand_choose=0.0):
    s_policy = {}
    if rand_choose == 0.0:
        for s, a_q in action_vals.iteritems():
            acts, qs = zip(*a_q.items())
            max_q = max(qs)
            max_acts = [acts[i] for i, qv in enumerate(qs) if qv == max_q]
            probs = [1/len(max_acts) for _ in max_acts]
            s_policy[s] = dict(zip(max_acts, probs))
    else:
        for s, a_q in action_vals.iteritems():
            acts, qs = zip(*a_q.items())
            max_q = max(qs)
            max_acts = [acts[i] for i, qv in enumerate(qs) if qv == max_q]
            nonmax_prob = rand_choose/len(acts)
            max_prob = (1-rand_choose)/len(max_acts) + nonmax_prob
            probs = [max_prob if a in max_acts else nonmax_prob for a in acts]
            s_policy[s] = dict(zip(acts, probs))
    return s_policy

def calc_egreedy_dist(action_vals, rand_choose=0.0):
    actions, qs = zip(*action_vals.items())
    max_q = max(qs)
    max_as = [a for a in actions if action_vals[a] == max_q]
    rand_p = rand_choose/len(actions)
    max_p = (1-rand_choose)/len(max_as)
    dist = {}
    for a in actions:
        if action_vals[a] == max_q:
            dist[a] = rand_p + max_p
        else:
            dist[a] = rand_p
    return dist


def sample_prob_dict(pdict):
    outs, p_s = zip(*pdict.items())
    out_i = np.random.choice(range(len(outs)), p=p_s)
    return outs[out_i]


def calc_traj_probability(policy, traj, get_log=False):
    if get_log:
        prob = 0
        for s, a in traj:
            prob += np.log(policy[s][a])
        return prob
    else:
        prob = 1
        for s, a in traj:
            prob *= policy[s][a]
        return prob


def argmax_dict(mydict, return_all_maxes=False, return_as_list=False):
    max_v = -np.inf
    max_k = []
    for k, v in mydict.iteritems():
        if v > max_v:
            max_v = v
            max_k = [k, ]
        elif v == max_v:
            max_k.append(k)

    if len(max_k) > 1:
        if return_all_maxes:
            return max_k
        else:
            if return_as_list:
                return [np.random.choice(max_k), ]
            else:
                return np.random.choice(max_k)
    else:
        if return_as_list:
            return max_k
        else:
            return max_k[0]
