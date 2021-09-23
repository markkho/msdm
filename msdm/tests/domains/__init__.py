from typing import Sequence

from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, DeterministicShortestPathProblem
from msdm.core.distributions import \
    Distribution, DictDistribution,\
    DeterministicDistribution, UniformDistribution
from msdm.tests.domains.russell_norvig import make_russell_norvig_grid

class GNTFig6_6(TabularMarkovDecisionProcess):
    T = [
        (((1, 2), 5), ((7,), 19), ((3, 9), 12)),
        (((4, 5), 4), ((11, 13), 4), ((6,), 2)),
        (((11,), 20), ((6,), 4)),
        (((6,), 8), ((8, 9), 5)),
        (((10,), 5), ((11, 12), 3)),
        (((11, 12), 4),),
        (((13, 14, 15), 5),),
        (((13,), 20), ((14, 15), 15)),
        (((14, 15), 6), ((9,), 4)),
        (((14, 15), 9),),
        (((5, 12), 7),),
        (((12,), 10), ((13, 14), 6)),
        (),
        (((14, 16), 35),),
        (((15, 16), 25),),
        (),
        (),
    ]

    '''
    Acyclic MDP from Ghallab, Nau, Traverso Figure 6.6
    '''
    def initial_state_dist(self) -> Distribution:
        return DeterministicDistribution(0)

    def is_terminal(self, s):
        return s in (12, 15, 16)

    def actions(self, s) -> Sequence:
        return [0, 1, 2]
        # dests = GNTFig6_6.T[s]
        # return [a for a in range(len(dests)) if dests[a][0]]

    def next_state_dist(self, s, a):
        if a < len(GNTFig6_6.T[s]):
            ns = GNTFig6_6.T[s][a][0]
        else:
            ns = [s]
        return UniformDistribution(ns)

    def reward(self, s, a, ns) -> float:
        if a < len(GNTFig6_6.T[s]):
            return -GNTFig6_6.T[s][a][1]
        return -100 # HACK

class Counter(DeterministicShortestPathProblem, TabularMarkovDecisionProcess):
    '''
    MDP where actions are increment/decrement and goal is to reach some count.
    '''
    def __init__(self, goal, *, initial_state=0):
        self._initial_state = initial_state
        self.goal = goal

    def initial_state(self):
        return self._initial_state

    def actions(self, s):
        return [1, -1]

    def next_state(self, s, a):
        ns = s + a
        if ns < 0 or self.goal < ns:
            return s
        return ns

    def reward(self, s, a, ns):
        return -1

    def is_terminal(self, s):
        return s == self.goal

class Geometric(TabularMarkovDecisionProcess):
    '''
    MDP where actions are to draw from a Bernoulli or wait.
    Goal is to get a 1 from the Bernoulli, which has probability `p`.
    '''
    def __init__(self, *, p=1/2):
        self.p = p

    def initial_state_dist(self):
        return DictDistribution({0: 1})

    def actions(self, s):
        return ['flip', 'wait']

    def next_state_dist(self, s, a):
        if a == 'wait':
            return DictDistribution({s: 1})
        elif a == 'flip':
            return DictDistribution({0: 1 - self.p, 1: self.p})

    def reward(self, s, a, ns):
        return -1

    def is_terminal(self, s):
        return s == 1

class VaryingActionNumber(DeterministicShortestPathProblem, TabularMarkovDecisionProcess):
    '''
    Counting MDP where actions at every state vary. Used to test handling of MDPs with
    varying numbers of states.
    '''
    def initial_state(self):
        return 0

    def actions(self, s):
        return {
            0: [+1],
            1: [-1, +1],
            2: [-1],
        }[s]

    def next_state(self, s, a):
        # Intentionally coding like this to ensure that
        # invalid access of this function results in a KeyError
        return {
            (0, +1): 1,
            (1, -1): 0,
            (1, +1): 2,
            (2, -1): 1,
        }[s, a]

    def reward(self, s, a, ns):
        return -1

    def is_terminal(self, s):
        return s == 2
