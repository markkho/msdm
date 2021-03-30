from scipy.special import softmax, logsumexp
import warnings
import numpy as np
from msdm.core.problemclasses.mdp import TabularPolicy, \
    TabularMarkovDecisionProcess, DeterministicTabularPolicy
from msdm.core.algorithmclasses import Plans, Result

class VectorizedValueIteration(Plans):
    def __init__(self,
                 iterations=None,
                 discount_rate=1.0,
                 entropy_regularization=False,
                 convergence_diff=1e-5,
                 temperature=1.0):
        self.iters = iterations
        self.dr = discount_rate
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
        terminal_sidx = np.where(1 - nt)[0]

        v = np.zeros(len(ss))
        for i in range(iterations):
            q = np.einsum("san,san->sa", tf, rf + self.dr * v[None, None, :])
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
            pi = softmax((1 / self.temp) * q, axis=-1)
        else:
            # This ensures we assign equal probability to actions that result
            # in the same q-values.
            pi = np.zeros_like(q)
            validq = q + aa
            pi[validq == np.max(validq, axis=-1, keepdims=True)] = 1
            pi /= pi.sum(axis=-1, keepdims=True)

        # create result object
        res = Result()
        if i == (iterations - 1):
            warnings.warn(f"VI not converged after {iterations} iterations")
            res.converged = False
        else:
            res.converged = True
        res.mdp = mdp
        cls = TabularPolicy if ((pi < 1) & (pi > 0)).any() else DeterministicTabularPolicy
        res.policy = res.pi = cls(mdp.state_list, mdp.action_list, policy_matrix=pi, mdp=mdp)
        res._valuevec = v
        vf = mdp.state_map()
        for s, vi in zip(mdp.state_list, v):
            vf[s] = vi
        res.valuefunc = res.V = vf
        res._qvaluemat = q
        res.iterations = i
        res.max_bellman_error = diff
        qf = mdp.state_action_map()
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                qf[s][a] = q[si, ai]
        res.actionvaluefunc = res.Q = qf
        return res
