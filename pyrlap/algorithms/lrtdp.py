from pyrlap.core.problemclasses.mdp import MarkovDecisionProcess
from pyrlap.core.assignment import DefaultAssignmentMap, \
    AssignmentMap
from pyrlap.core.algorithmclasses import Plans, Result

def iter_dist_prob(dist):
    '''
    Iterate over support of a distribution and probability.
    Useful for computing expectations.
    '''
    for e in dist.support:
        yield e, dist.prob(e)

class LRTDP(Plans):
    '''
    Labeled Real-Time Dynamic Programming (Bonet & Geffner 2003)

    Implementation mixes names from Bonet & Geffner 2003 and Ghallab, Nau, Traverso.
    '''
    def __init__(self,
                 error_margin=1e-2,
                 heuristic=None,
                 iterations=int(2**30)
                 ):
        self.error_margin = error_margin
        self.heuristic = heuristic
        self.iterations = iterations

    def planOn(self, mdp: MarkovDecisionProcess):
        self.res = Result()
        self.lrtdp(
            mdp, heuristic=self.heuristic, iterations=self.iterations
        )
        res = self.res
        res.policy = AssignmentMap()
        res.Q = AssignmentMap()
        for s in mdp.states:
            res.policy[s] = self.policy(mdp, s)
            res.Q[s] = AssignmentMap()
            for a in mdp.actions:
                res.Q[s][a] = self.Q(mdp, s, a)

        #clear result
        self.res = None
        return res

    def _bellman_update(self, mdp, s):
        '''
        Following Bonet & Geffner 2003, we only explicitly store value
        and compute Q values and the policy from it, important in computing
        the residual in _check_solved().
        '''
        self.res.V[s] = max(self.Q(mdp, s, a) for a in mdp.actions)

    def Q(self, mdp, s, a):
        q = 0
        for ns, prob in iter_dist_prob(mdp.getNextStateDist(s, a)):
            future = 0
            if not mdp.isTerminal(ns):
                future = self.res.V[ns]
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

        self.res.V = DefaultAssignmentMap(heuristic)

        # Keeping track of "labels": which states have been solved
        self.res.solved = DefaultAssignmentMap(lambda: False)
        for s in mdp.states:
            # Terminal states are solved.
            if mdp.isTerminal(s):
                self.res.solved[s] = True

        for _ in range(iterations):
            if all(self.res.solved[s] for s in mdp.getInitialStateDist().support):
                return
            self.lrtdp_trial(mdp, mdp.getInitialStateDist().sample())

    def lrtdp_trial(self, mdp, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = []
        while not self.res.solved[s]:
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
        if not self.res.solved[s]:
            open.append(s)
        while open:
            s = open.pop()
            closed.append(s)
            residual = self.res.V[s] - self.Q(mdp, s, self.policy(mdp, s))
            if abs(residual) > self.error_margin:
                flag = False
            else:
                for ns in mdp.getNextStateDist(s, self.policy(mdp, s)).support:
                    if not self.res.solved[ns] and ns not in open and ns not in closed:
                        open.append(ns)
        if flag:
            for ns in closed:
                self.res.solved[ns] = True
        else:
            while closed:
                s = closed.pop()
                self._bellman_update(mdp, s)
        return flag
