from __future__ import division
import logging
import warnings
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)


SANSRTuple = namedtuple('SANSRTuple', 's a ns r')
SANSTuple = namedtuple('SANSTuple', 's a ns')
SATuple = namedtuple('SATuple', 's a')

#================================#
#
#    Functions related to policies
#    and probabilities
#
#================================#
np.seterr(all='raise')

def calc_softmax_dist(action_vals, temp=1.0):
    #normalization trick
    mval = max(action_vals.values())
    action_vals = {a: v - mval for a, v in action_vals.items()}

    aweights = {}
    norm = 0
    for a, q in action_vals.items():
        # handle underflow
        try:
            p = np.exp(q/temp)
        except FloatingPointError:
            p = 0
            warnings.warn(("Softmax underflow (q = %g, temp = %g); " +
                          "setting prob to 0.") % (q, temp))
        norm += p
        aweights[a] = p

    aprobs = {}
    for a, p in aweights.items():
        #handle underflow
        try:
            p = p/norm
        except FloatingPointError:
            p = 0
            warnings.warn(("Softmax underflow (q = %g, temp = %g); " +
                           "setting prob to 0.") % (p, temp))
        aprobs[a] = p
    return aprobs

def calc_softmax_policy(stateaction_vals, temp=1):
    soft_max_policy = {}
    for s, a_q in stateaction_vals.items():
        soft_max_policy[s] = calc_softmax_dist(a_q, temp=temp)
    return soft_max_policy

def calc_esoftmax_dist(a_vals, temp=0.0, randchoose=0.0):
    """
    See work by Nassar & Frank (2016) and Collins and Frank (2018)

    http://ski.clps.brown.edu/papers/NassarFrank_curopin.pdf and
    http://ski.clps.brown.edu/papers/CollinsFrank_PNAS_supp.pdf
    """
    if len(a_vals) == 1:
        return {list(a_vals.keys())[0]: 1}

    if temp == 0.0:
        maxval = max(a_vals.values())
        maxacts = [a for a, v in a_vals.items() if v == maxval]
        act_randchoose = randchoose/len(a_vals)
        act_maxchoose = (1-randchoose)/len(maxacts)
        a_p = {}
        for a in a_vals.keys():
            a_p[a] = act_randchoose
            if a in maxacts:
                a_p[a] += act_maxchoose
    else:
        sm = calc_softmax_dist(a_vals, temp)
        act_randchoose = randchoose/len(a_vals)
        a_p = {}
        for a, smp in sm.items():
            #handles underflow
            try:
                rand_smp = (1 - randchoose)*smp
            except FloatingPointError:
                rand_smp = 0
            a_p[a] = act_randchoose + rand_smp
    return a_p

def calc_esoftmax_policy(sa_vals, temp=0.0, randchoose=0.0):
    policy = {}
    for s, a_q in sa_vals.items():
        policy[s] = calc_esoftmax_dist(a_q, temp=temp, randchoose=randchoose)
    return policy

def calc_stochastic_policy(sa_vals, rand_choose=0.0):
    return calc_esoftmax_policy(sa_vals, temp=0.0, randchoose=rand_choose)

def calc_egreedy_dist(action_vals, rand_choose=0.0):
    return calc_esoftmax_dist(action_vals, randchoose=rand_choose, temp=0.0)

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

def argmax_dict(mydict,
                return_one=True):
    max_v = max(mydict.values())
    max_k = [k for k, v in mydict.items() if v == max_v]

    if return_one:
        if len(max_k) == 0:
            return max_k[0]
        else:
            return np.random.choice(max_k)
    else:
        return max_k

def max_index(mylist, return_one=True):
    max_val = max(mylist)
    max_i = [i for i, v in enumerate(mylist) if v == max_val]
    if return_one:
        if len(max_i) == 0:
            return max_i[0]
        else:
            return np.random.choice(max_i)
    else:
        return max_i

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)