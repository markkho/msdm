import numpy as np
import copy
from msdm.core.problemclasses.mdp import MarkovDecisionProcess
from msdm.core.problemclasses.mdp.policy.partialpolicy import PartialPolicy
from msdm.core.algorithmclasses import Plans, Result

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
                 iterations=int(2**30),
                 randomize_action_order=False,
                 max_trial_length=None,
                 seed=None
                 ):
        self.error_margin = error_margin
        self.heuristic = heuristic
        self.iterations = iterations
        self.randomize_action_order = randomize_action_order
        self.seed = seed
        if max_trial_length is None:
            max_trial_length = float('inf')
        self.max_trial_length = max_trial_length

    def plan_on(self, mdp: MarkovDecisionProcess):
        self.res = Result()
        if self.seed is None:
            self.res.seed = np.random.randint(int(2**30))
        else:
            self.res.seed = self.seed
        np.random.seed(self.res.seed)
        self.lrtdp(
            mdp, heuristic=self.heuristic, iterations=self.iterations
        )
        res = self.res
        res.policy = mdp.state_action_map()
        res.Q = mdp.state_action_map()
        for s in sum(self.res.trials, []) + [state for state, solved in res.solved.items() if solved]:
            if s in res.policy:
                continue

            # res.policy[s] = HashDictionary(mdp.hash_action)
            res.policy[s][self.policy(mdp, s)] = 1
            # res.Q[s] = HashDictionary(mdp.hash_action)
            for a in mdp.actions(s):
                res.Q[s][a] = self.Q(mdp, s, a)
        res.policy = PartialPolicy(res.policy)

        #clear result
        self.res = None
        return res

    def lrtdp(self, mdp, heuristic=None, iterations=None):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        if heuristic is None:
            heuristic = lambda s: 0

        self.res.V = mdp.state_map(default_value=heuristic)
        self.res.action_orders = mdp.state_map()

        self.res.trials = []
        self.res.trials_solved = []

        # Keeping track of "labels": which states have been solved
        self.res.solved = mdp.state_map(default_value=lambda: False)

        for _ in range(iterations):
            if all(self.res.solved[s] for s in mdp.initial_state_dist().support):
                return
            self.lrtdp_trial(mdp, mdp.initial_state_dist().sample())

    def lrtdp_trial(self, mdp, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = [s, ]
        while not self.res.solved[s]:
            self._bellman_update(mdp, s)
            s = mdp.next_state_dist(s, self.policy(mdp, s)).sample()
            visited.append(s)

            # Terminal states are solved.
            if mdp.is_terminal(s):
                self.res.solved[s] = True
            if len(visited) > self.max_trial_length:
                break
        self.res.trials.append(copy.deepcopy(visited))
        self.res.trials_solved.append(copy.deepcopy(self.res.solved))
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
                for ns in mdp.next_state_dist(s, self.policy(mdp, s)).support:
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

    def _bellman_update(self, mdp, s):
        '''
        Following Bonet & Geffner 2003, we only explicitly store value
        and compute Q values and the policy from it, important in computing
        the residual in _check_solved().
        '''
        self.res.V[s] = max(self.Q(mdp, s, a) for a in mdp.actions(s))

    def Q(self, mdp, s, a):
        if mdp.is_terminal(s):
            return 0
        q = 0
        for ns, prob in iter_dist_prob(mdp.next_state_dist(s, a)):
            future = 0
            if not mdp.is_terminal(ns):
                future = self.res.V[ns]
            q += prob * (mdp.reward(s, a, ns) + future)
        return q

    def policy(self, mdp, s):
        if s in self.res.action_orders:
            action_list = self.res.action_orders[s]
        else:
            if self.randomize_action_order:
                aa = mdp.actions(s)
                action_list = [aa[i] for i in np.random.permutation(len(aa))]
            else:
                action_list = mdp.actions(s)
            self.res.action_orders[s] = action_list
        return max(action_list, key=lambda a: self.Q(mdp, s, a))

    def expected_one_step_reward_heuristic(self, mdp):
        '''
        This admissible heuristic is a generally applicable one.
        The heuristic value for a state is the best one-step reward.
        '''
        return lambda s: 0 if mdp.is_terminal(s) else max(
            sum(
                prob * mdp.reward(s, a, ns)
                for ns, prob in iter_dist_prob(mdp.next_state_dist(s, a))
            )
            for a in mdp.action_dist(s).support
        )

