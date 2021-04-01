from scipy.special import softmax, logsumexp
import warnings
import numpy as np
from collections import defaultdict
from msdm.core.problemclasses.mdp import \
    TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

class VectorizedValueIteration(Plans):
    def __init__(self,
                 iterations=None,
                 entropy_regularization=False,
                 convergence_diff=1e-5,
                 temperature=1.0):
        self.iters = iterations
        self.entreg = entropy_regularization
        self.temp = temperature
        self._policy = None
        self.convergence_diff = convergence_diff

    def plan_on(self, mdp: TabularMarkovDecisionProcess):

        ss = mdp.state_list
        tf = mdp.transition_matrix
        rf = mdp.reward_matrix
        nt = mdp.nonterminal_state_vec
        rs = mdp.reachable_state_vec
        am = mdp.action_matrix

        iterations = self.iters if self.iters is not None else max(len(ss), int(1e5))

        #available actions - add -inf if it is not available
        aa = am.copy()
        assert np.all(aa[np.nonzero(aa)] == 1) # If this is true, then the next line is unnecessary.
        aa[np.nonzero(aa)] = 1
        aa = np.log(aa)
        assert (aa == np.log(am)).all() # if this is true, then we should delete `aa` and just use np.log(am) in place
        terminal_sidx = np.where(1 - nt)[0]

        v = np.zeros(len(ss))
        for i in range(iterations):
            q = np.einsum("san,san->sa", tf, rf + mdp.discount_rate * v[None, None, :])
            if self.entreg:
                nv = self.temp * logsumexp((1 / self.temp) * q + np.log(am),
                                          axis=-1)
                nv[terminal_sidx] = 0 #terminal states are always 0 reward
            else:
                nv = np.max(q + aa, axis=-1)
                nv[terminal_sidx] = 0 #terminal states are always 0 reward

            diff = (v - nv)*rs
            if np.abs(diff).max() < self.convergence_diff:
                break
            v = nv

        if self.entreg:
            pi = softmax((1 / self.temp) * q + np.log(am), axis=-1)
            pi = TabularPolicy.from_matrix(mdp.state_list, mdp.action_list, pi)
        else:
            validq = q + aa
            pi = TabularPolicy.from_q_matrix(mdp.state_list, mdp.action_list, validq)

        # create result object
        res = PlanningResult()
        if i == (iterations - 1):
            warnings.warn(f"VI not converged after {iterations} iterations")
            res.converged = False
        else:
            res.converged = True
        res.mdp = mdp
        res.policy = res.pi = pi
        res._valuevec = v
        vf = dict()
        for s, vi in zip(mdp.state_list, v):
            vf[s] = vi
        res.valuefunc = res.V = vf
        res._qvaluemat = q
        res.iterations = i
        res.max_bellman_error = diff
        qf = defaultdict(lambda : dict())
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                qf[s][a] = q[si, ai]
        res.actionvaluefunc = res.Q = qf
        return res
