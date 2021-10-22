from scipy.special import softmax, logsumexp
import warnings
import numpy as np
from collections import defaultdict
from msdm.core.problemclasses.mdp import \
    TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

class ValueIteration(Plans):
    def __init__(self,
                 iterations=None,
                 convergence_diff=1e-5,
                 check_unreachable_convergence=True
                 ):
        self.iterations = iterations
        self.convergence_diff = convergence_diff
        self.check_unreachable_convergence = check_unreachable_convergence

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        ss = mdp.state_list
        tf = mdp.transition_matrix
        rf = mdp.reward_matrix
        nt = mdp.nonterminal_state_vec
        rs = mdp.reachable_state_vec
        am = mdp.action_matrix

        # transition function goes nowhere
        tf = tf*nt[:, None, None]
        # reward function assigns 0 to all transitions out of a terminal
        rf = rf*nt[:, None, None]

        iterations = self.iterations
        if iterations is None:
            iterations = max(len(ss), int(1e5))

        v = np.zeros(len(ss))
        for i in range(iterations):
            q = np.einsum("san,san->sa", tf, rf + mdp.discount_rate * v[None, None, :])
            nv = np.max(q + np.log(am), axis=-1)
            if self.check_unreachable_convergence:
                diff = (v - nv)
            else:
                diff = (v - nv)*rs
            if np.abs(diff).max() < self.convergence_diff:
                break
            v = nv

        validq = q + np.log(am)
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
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])
        return res

# for backward compatibility
VectorizedValueIteration = ValueIteration
