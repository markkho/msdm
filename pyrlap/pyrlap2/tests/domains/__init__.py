from pyrlap.pyrlap2.core.problemclasses.mdp import TabularMarkovDecisionProcess
from pyrlap.pyrlap2.core.distributions import Multinomial, Distribution

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
    def getInitialStateDist(self) -> Distribution:
        return Multinomial([0])

    def isTerminal(self, s):
        return s in (12, 15, 16)

    def getActionDist(self, s) -> Distribution:
        dests = GNTFig6_6.T[s]
        return Multinomial([
            a for a in range(len(dests)) if dests[a][0]])

    def getNextStateDist(self, s, a):
        if a < len(GNTFig6_6.T[s]):
            ns = GNTFig6_6.T[s][a][0]
        else:
            ns = [s]
        return Multinomial(ns)

    def getReward(self, s, a, ns) -> float:
        if a < len(GNTFig6_6.T[s]):
            return -GNTFig6_6.T[s][a][1]
        return -100 # HACK
