from typing import Iterable
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.distributions import DiscreteFactorTable

class StickyActionMDP(TabularMarkovDecisionProcess):
    def __init__(self, basemdp, initAction=None, switchCost=-1):
        super().__init__()
        self.mdp = basemdp
        self.switchCost = switchCost
        self.initAction = initAction
    
    def is_terminal(self, s):
        return self.mdp.is_terminal(s['groundState'])
    
    def next_state_dist(self, s, a) -> DiscreteFactorTable:
        nsDist = self.mdp.next_state_dist(s['groundState'], a)
        nsDist = DiscreteFactorTable(
            [{'groundState': ns, 'curAction': a} for ns in nsDist.support],
            logits=nsDist.logits
        )
        return nsDist
    
    def reward(self, s, a, ns) -> float:
        r = self.mdp.reward(s['groundState'], a, ns['groundState'])
        if s['curAction'] != a:
            r += self.switchCost
        return r

    def actions(self, s) -> Iterable:
        return self.mdp.actions(s['groundState'])

    def initial_state_dist(self) -> DiscreteFactorTable:
        S0 = self.mdp.initial_state_dist()
        if self.initAction is None:
            SA0 = DiscreteFactorTable([])
            for s0 in S0.support:
                s0actions = self.mdp.actions(s0) #actions for initial state
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