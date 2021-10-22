import warnings
import numpy as np
from collections import defaultdict
from msdm.core.problemclasses.mdp import \
    TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

class PolicyIteration(Plans):
    def __init__(self, iterations=None,
                 check_unreachable_convergence=True):
        self.iterations = iterations
        self.check_unreachable_convergence = check_unreachable_convergence

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        ss = mdp.state_list
        tf = mdp.transition_matrix
        rf = mdp.reward_matrix
        nt = mdp.nonterminal_state_vec
        rs = mdp.reachable_state_vec
        am = mdp.action_matrix

        # In general, terminal states only lead to themselves
        # and return zero rewards, but to ensure that the
        # the transition matrix is non-singular
        # we assume terminal states transition nowhere.
        tf = tf*nt[:, None, None]

        # reward function assigns 0 to all transitions out of a terminal
        rf = rf*nt[:, None, None]

        iterations = self.iterations
        if iterations is None:
            iterations = max(len(ss), int(1e5))

        # Initialize to uniform random policy over available
        # actions.
        pi = am / am.sum(axis=1, keepdims=True)

        for i in range(iterations):
            # Calculate the expected per-state reward under
            # the current policy.
            s_rf = (pi[:, :, None] * tf[:, :, :] * rf[:, :, :]).sum(axis=(1, 2))

            # Construct a markov chain, marginalizing over
            # the current policy. Only consider reachable states.
            mp = (pi[:, :, None] * tf[:, :, :]).sum(axis=1)
            mp = rs[:, None] * mp

            # The value the solution to a set of linear equations.
            v = np.linalg.solve(np.eye(len(ss)) - mdp.discount_rate * mp, s_rf)

            # The action value is the expectation over next state transitions
            q = (tf[:, :, :] * (rf[:, :, :] + mdp.discount_rate * v[None, None, :])).sum(axis=2)

            # Calculate the new policy, taking into account
            # the "infinite" cost of unavailable actions.
            new_pi = np.zeros_like(pi)
            np.put_along_axis(new_pi, (q + np.log(am)).argmax(axis=1)[:, None], values=1, axis=1)

            # Check convergence
            if self.check_unreachable_convergence:
                converged = (new_pi == pi).all()
            else:
                converged = (new_pi[rs, :] == pi[rs, :]).all()
            if converged:
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
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])
        return res
