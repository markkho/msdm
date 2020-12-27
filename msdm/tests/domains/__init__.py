from typing import Iterable

from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.distributions import DiscreteFactorTable, Distribution

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
        return DiscreteFactorTable([0])

    def is_terminal(self, s):
        return s in (12, 15, 16)

    def actions(self, s) -> Iterable:
        return [0, 1, 2]
        # dests = GNTFig6_6.T[s]
        # return [a for a in range(len(dests)) if dests[a][0]]

    def next_state_dist(self, s, a):
        if a < len(GNTFig6_6.T[s]):
            ns = GNTFig6_6.T[s][a][0]
        else:
            ns = [s]
        return DiscreteFactorTable(ns)

    def reward(self, s, a, ns) -> float:
        if a < len(GNTFig6_6.T[s]):
            return -GNTFig6_6.T[s][a][1]
        return -100 # HACK

class CountToTen(TabularMarkovDecisionProcess):
    def is_terminal(self, s):
        return s == 10

    def next_state_dist(self, s, a):
        if s == 10:
            return DiscreteFactorTable([])
        return DiscreteFactorTable([s+a])

    def actions(self, s):
        if s < 0:
            return DiscreteFactorTable([1])
        if s < 5:
            return DiscreteFactorTable([1, -1])
        return DiscreteFactorTable([1])

    def initial_state_dist(self):
        return DiscreteFactorTable([0,])

    def reward(self, s, a, ns):
        return -1

