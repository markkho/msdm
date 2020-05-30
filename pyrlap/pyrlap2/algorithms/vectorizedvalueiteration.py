from scipy.special import softmax, logsumexp
import numpy as np
from pyrlap.pyrlap2.core import Planner, TabularAgent, \
    TabularMarkovDecisionProcess, Multinomial


class VectorizedValueIteration(Planner, TabularAgent):
    def __init__(self,
                 iterations=50,
                 discountRate=1.0,
                 entropyRegularization=False,
                 temperature=1.0):
        self.iters = iterations
        self.dr = discountRate
        self.entreg = entropyRegularization
        self.temp = temperature

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
        self._policymat = pi
        self._valuevec = v
        self._qvaluemat = q
        self._states = ss
        self._actions = mdp.actions

    def getActionDist(self, state):
        si = self.states.index(state)
        ap = self.policymat[si]
        return Multinomial(support=self.actions, probs=ap)

    @property
    def valuefunc(self):
        vf = dict(zip(self.states, self.valuevec))
        return vf

    @property
    def actionvaluefunc(self):
        qf = {}
        for si, s in enumerate(self.states):
            qf[s] = {}
            for ai, a in enumerate(self.actions):
                qf[s][a] = self.qvaluemat[si, ai]
        return qf

    @property
    def policy(self):
        pi = {}
        for si, s in enumerate(self.states):
            pi[s] = {}
            for ai, a in enumerate(self.actions):
                pi[s][a] = self.policymat[si, ai]
        return pi