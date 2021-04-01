import warnings
import numpy as np
from collections import defaultdict
from msdm.core.problemclasses.mdp import \
    TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

class PolicyIteration(Plans):
    def __init__(self, iterations=None):
        self.iterations = iterations

    def __call__(self, mdp: TabularMarkovDecisionProcess):
        return self.plan_on(mdp)

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        ss = mdp.state_list
        s0 = mdp.initial_state_vec
        tf = mdp.transition_matrix
        rf = mdp.reward_matrix
        nt = mdp.nonterminal_state_vec
        rs = mdp.reachable_state_vec
        am = mdp.action_matrix

        iterations = self.iterations
        if iterations is None:
            iterations = max(len(ss), int(1e5))

        pi = np.ones(tf.shape[:-1])
        pi = pi / pi.sum(axis=1, keepdims=True)

        for i in range(iterations):
            s_rf = (pi[:, :, None] * tf[:, :, :] * rf[:, :, :]).sum(axis=(1, 2))
            mp = (rs[:, None] * (pi[:, :, None] * tf[:, :, :]).sum(axis=1) * nt[None,:])
            v = np.linalg.solve(np.eye(len(ss)) - mdp.discount_rate * mp, s_rf)
            q = (tf[:, :, :] * (rf[:, :, :] + v[None, None, :])).sum(axis=2)

            new_pi = np.zeros_like(pi)
            np.put_along_axis(new_pi, (q + np.log(am)).argmax(axis=1)[:, None], values=1, axis=1)
            if (new_pi[rs, :] == pi[rs, :]).all():
                break
            pi = new_pi

        validq = q + np.log(am)
        pi = TabularPolicy.from_q_matrix(mdp.state_list, mdp.action_list, validq)

        # create result object
        res = PlanningResult()
        if i == (iterations - 1):
            warnings.warn(f"Policy Iteration not converged after {iterations} iterations")
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
        qf = defaultdict(lambda : dict())
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                qf[s][a] = q[si, ai]
        res.actionvaluefunc = res.Q = qf
        # res.initial_value = s0 @ v
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])
        return res