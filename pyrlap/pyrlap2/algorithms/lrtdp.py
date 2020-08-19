from pyrlap.pyrlap2.core import MarkovDecisionProcess
from pyrlap.pyrlap2.core import DefaultAssignmentMap

def iter_dist_prob(dist):
    '''
    Iterate over support of a distribution and probability.
    Useful for computing expectations.
    '''
    for e in dist.support:
        yield e, dist.prob(e)

class LRTDP(object):
    '''
    Labeled Real-Time Dynamic Programming (Bonet & Geffner 2003)

    Implementation mixes names from Bonet & Geffner 2003 and Ghallab, Nau, Traverso.
    '''
    def __init__(self, error_margin=1e-2):
        self.error_margin = error_margin

    def planOn(self, mdp: MarkovDecisionProcess, heuristic=None, iterations=int(2**30)):
        return self.lrtdp(mdp, heuristic=heuristic, iterations=iterations)

    def _bellman_update(self, mdp, s):
        '''
        Following Bonet & Geffner 2003, we only explicitly store value
        and compute Q values and the policy from it, important in computing
        the residual in _check_solved().
        '''
        self.V[s] = max(self.Q(mdp, s, a) for a in mdp.actions)

    def Q(self, mdp, s, a):
        q = 0
        for ns, prob in iter_dist_prob(mdp.getNextStateDist(s, a)):
            future = 0
            if not mdp.isTerminal(ns):
                future = self.V[ns]
            q += prob * (mdp.getReward(s, a, ns) + future)
        return q

    def policy(self, mdp, s):
        return max(mdp.actions, key=lambda a: self.Q(mdp, s, a))

    def expected_one_step_reward_heuristic(self, mdp):
        '''
        This admissible heuristic is a generally applicable one.
        The heuristic value for a state is the best one-step reward.
        '''
        return lambda s: 0 if mdp.isTerminal(s) else max(
            sum(
                prob * mdp.getReward(s, a, ns)
                for ns, prob in iter_dist_prob(mdp.getNextStateDist(s, a))
            )
            for a in mdp.getActionDist(s).support
        )

    def lrtdp(self, mdp, heuristic=None, iterations=None):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        if heuristic is None:
            heuristic = lambda s: 0

        self.V = DefaultAssignmentMap(heuristic)

        # Keeping track of "labels": which states have been solved
        self.solved = DefaultAssignmentMap(lambda: False)
        for s in mdp.states:
            # Terminal states are solved.
            if mdp.isTerminal(s):
                self.solved[s] = True

        for _ in range(iterations):
            if all(self.solved[s] for s in mdp.getInitialStateDist().support):
                return
            self.lrtdp_trial(mdp, mdp.getInitialStateDist().sample())

    def lrtdp_trial(self, mdp, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = []
        while not self.solved[s]:
            visited.append(s)
            self._bellman_update(mdp, s)
            s = mdp.getNextStateDist(s, self.policy(mdp, s)).sample()
        s = visited.pop()
        while self._check_solved(mdp, s) and visited:
            s = visited.pop()

    def _check_solved(self, mdp, s):
        # GNT Algorithm 6.18
        flag = True
        open = []
        closed = []
        if not self.solved[s]:
            open.append(s)
        while open:
            s = open.pop()
            closed.append(s)
            residual = self.V[s] - self.Q(mdp, s, self.policy(mdp, s))
            if abs(residual) > self.error_margin:
                flag = False
            else:
                for ns in mdp.getNextStateDist(s, self.policy(mdp, s)).support:
                    if not self.solved[ns] and ns not in open and ns not in closed:
                        open.append(ns)
        if flag:
            for ns in closed:
                self.solved[ns] = True
        else:
            while closed:
                s = closed.pop()
                self._bellman_update(mdp, s)
        return flag
