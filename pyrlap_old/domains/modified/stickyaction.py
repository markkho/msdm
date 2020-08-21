from typing import Iterable
from pyrlap_old.pyrlap2.core.problemclasses.mdp import TabularMarkovDecisionProcess
from pyrlap_old.pyrlap2.core.distributions import DiscreteFactorTable

class StickyActionMDP(TabularMarkovDecisionProcess):
    def __init__(self, basemdp, initAction=None, switchCost=-1):
        super().__init__()
        self.mdp = basemdp
        self.switchCost = switchCost
        self.initAction = initAction
    
    def isTerminal(self, s):
        return self.mdp.isTerminal(s['groundState'])
    
    def getNextStateDist(self, s, a) -> DiscreteFactorTable:
        nsDist = self.mdp.getNextStateDist(s['groundState'], a)
        nsDist = DiscreteFactorTable(
            [{'groundState': ns, 'curAction': a} for ns in nsDist.support],
            logits=nsDist.logits
        )
        return nsDist
    
    def getReward(self, s, a, ns) -> float:
        r = self.mdp.getReward(s['groundState'], a, ns['groundState'])
        if s['curAction'] != a:
            r += self.switchCost
        return r

    def getActions(self, state) -> Iterable:
        return self.mdp.getActions(state['groundState'])

    def getInitialStateDist(self) -> DiscreteFactorTable:
        S0 = self.mdp.getInitialStateDist()
        if self.initAction is None:
            SA0 = DiscreteFactorTable([])
            for s0 in S0.support:
                s0actions = self.mdp.getActions(s0) #actions for initial state
                s0A0 = DiscreteFactorTable(
                    [{'groundState': s0, 'curAction': a} for a in s0actions]
                )
                SA0 = SA0 | s0A0
        else:
            SA0 = DiscreteFactorTable([{'curAction':self.initAction},])
        S0 = DiscreteFactorTable(
            support=[{'groundState': s} for s in S0.support],
            logits=S0.logits
        )
        return SA0 & S0