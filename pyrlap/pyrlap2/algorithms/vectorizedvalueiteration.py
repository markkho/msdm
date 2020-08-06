from scipy.special import softmax, logsumexp
from typing import Mapping
import numpy as np
from pyrlap.pyrlap2.core import TabularPolicy, \
    TabularMarkovDecisionProcess, Multinomial, AssignmentMap

class VectorizedValueIteration:
    def __init__(self,
                 iterations=50,
                 discountRate=1.0,
                 entropyRegularization=False,
                 temperature=1.0):
        self.iters = iterations
        self.dr = discountRate
        self.entreg = entropyRegularization
        self.temp = temperature
        self._policy = None

    def planOn(self, mdp: TabularMarkovDecisionProcess):
        ss = mdp.states
        tf = mdp.transitionmatrix
        rf = mdp.rewardmatrix
        nt = mdp.nonterminalstatevec
        am = mdp.actionmatrix
        v = np.zeros(len(ss))
        for i in range(self.iters):
            q = np.einsum("san,san->sa", tf, rf + self.dr * v)
            if self.entreg:
                v = self.temp * logsumexp((1 / self.temp) * q + np.log(am),
                                          axis=-1) * nt
            else:
                v = np.max(q, axis=-1)
        if self.entreg:
            pi = softmax((1 / self.temp) * q, axis=-1)
        else:
            pi = np.log(np.zeros_like(q))
            pi[q == np.max(q, axis=-1, keepdims=True)] = 1
            pi = softmax(pi, axis=-1)
        self._policy = TabularPolicy(mdp.states, mdp.actions, policymatrix=pi)
        self.states = mdp.states
        self.actions = mdp.actions
        self._valuevec = v
        self._qvaluemat = q

    @property
    def valuefunc(self) -> Mapping:
        vf = AssignmentMap()
        for s, v in zip(self.states, self._valuevec):
            vf[s] = v
        return vf

    @property
    def actionvaluefunc(self) -> Mapping:
        qf = AssignmentMap()
        for si, s in enumerate(self.states):
            qf[s] = AssignmentMap()
            for ai, a in enumerate(self.actions):
                qf[s][a] = self._qvaluemat[si, ai]
        return qf

    @property
    def policy(self) -> TabularPolicy:
        return self._policy

    #shortcuts
    @property
    def V(self):
        return self.valuefunc

    @property
    def Q(self):
        return self.actionvaluefunc
 
    @property
    def pi(self):
        return self.policy
