from scipy.special import softmax, logsumexp
import numpy as np
from msdm.core.problemclasses.mdp import TabularPolicy, \
    TabularMarkovDecisionProcess, DeterministicTabularPolicy
from msdm.core.algorithmclasses import Plans, Result
from msdm.core.assignment import AssignmentMap

class VectorizedValueIteration(Plans):
    def __init__(self,
                 iterations=50,
                 discount_rate=1.0,
                 entropy_regularization=False,
                 temperature=1.0):
        self.iters = iterations
        self.dr = discount_rate
        self.entreg = entropy_regularization
        self.temp = temperature
        self._policy = None

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        ss = mdp.state_list
        tf = mdp.transition_matrix
        rf = mdp.reward_matrix
        nt = mdp.nonterminal_state_vec
        am = mdp.action_matrix

        #available actions - add -inf if it is not available
        aa = am.copy()
        aa[np.nonzero(aa)] = 1
        aa = np.log(aa)
        terminal_sidx = np.where(1 - nt)[0]

        v = np.zeros(len(ss))
        for i in range(self.iters):
            q = np.einsum("san,san->sa", tf, rf + self.dr * v)
            if self.entreg:
                v = self.temp * logsumexp((1 / self.temp) * q + np.log(am),
                                          axis=-1)
                v[terminal_sidx] = 0 #terminal states are always 0 reward
            else:
                v = np.max(q + aa, axis=-1)
                v[terminal_sidx] = 0 #terminal states are always 0 reward
        if self.entreg:
            pi = softmax((1 / self.temp) * q, axis=-1)
        else:
            pi = np.log(np.zeros_like(q))
            pi[q == np.max(q, axis=-1, keepdims=True)] = 1
            pi = softmax(pi, axis=-1)

        # create result object
        res = Result()
        res.mdp = mdp
        cls = TabularPolicy if self.entreg else DeterministicTabularPolicy
        res.policy = res.pi = cls(mdp.state_list, mdp.action_list, policy_matrix=pi)
        res._valuevec = v
        vf = AssignmentMap([(s, vi) for s, vi in zip(mdp.state_list, v)])
        res.valuefunc = res.V = vf
        res._qvaluemat = q
        res.iterations = i
        qf = AssignmentMap()
        for si, s in enumerate(mdp.state_list):
            qf[s] = AssignmentMap()
            for ai, a in enumerate(mdp.action_list):
                qf[s][a] = q[si, ai]
        res.actionvaluefunc = res.Q = qf
        return res

    # def getSoftPolicy(self, temp=1.0):
    #     return TabularPolicy(
    #         self.states,
    #         self.actions,
    #         policymatrix=softmax(self._qvaluemat/temp, axis=-1)
    #     )
