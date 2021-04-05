import random
import copy
import warnings
from collections import defaultdict
from msdm.core.utils.dictutils import defaultdict2
from msdm.core.problemclasses.mdp import MarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

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
        self.res = PlanningResult()
        if self.seed is None:
            self.res.seed = random.randint(0, int(2**30))
        else:
            self.res.seed = self.seed
        self.rng = random.Random(self.seed)
        self.lrtdp(
            mdp, heuristic=self.heuristic, iterations=self.iterations
        )
        res = self.res
        res.policy = {}
        res.Q = defaultdict(lambda : dict())
        for s in sum(self.res.trials, []) + [state for state, solved in res.solved.items() if solved]:
            if s in res.policy:
                continue
            res.policy[s] = self.policy(mdp, s)
            for a in mdp.actions(s):
                res.Q[s][a] = self.Q(mdp, s, a)
        res.policy = TabularPolicy.from_deterministic_map(res.policy)
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])

        #clear result
        self.res = None
        return res

    def lrtdp(self, mdp, heuristic=None, iterations=None):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        if heuristic is None:
            heuristic = lambda s: 0

        self.res.V = defaultdict2(heuristic)
        self.res.action_orders = dict()

        self.res.trials = []
        self.res.trials_solved = []

        # Keeping track of "labels": which states have been solved
        self.res.solved = defaultdict2(lambda s: False)

        for i in range(iterations):
            if all(self.res.solved[s] for s in mdp.initial_state_dist().support):
                return
            self.lrtdp_trial(mdp, mdp.initial_state_dist().sample(rng=self.rng))
        if i == (iterations - 1):
            warnings.warn(f"LRTDP not converged after {iterations} iterations")
            self.res.converged = False
        else:
            self.res.converged = True

    def lrtdp_trial(self, mdp, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = [s, ]
        while not self.res.solved[s]:
            self._bellman_update(mdp, s)
            s = mdp.next_state_dist(s, self.policy(mdp, s)).sample(rng=self.rng)
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
        for ns, prob in mdp.next_state_dist(s, a).items():
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
                action_list = list(mdp.actions(s))
                self.rng.shuffle(action_list)
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
                for ns, prob in mdp.next_state_dist(s, a).items()
            )
            for a in mdp.action_dist(s).support
        )

