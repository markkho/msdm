from pyrlap.pyrlap2.core import TabularMarkovDecisionProcess
from pyrlap.pyrlap2.core import Multinomial

class StickyActionMDP(TabularMarkovDecisionProcess):
    def __init__(self, basemdp, initAction=None, initActionDist=None, switchCost=-1):
        super().__init__()
        self.mdp = basemdp
        self.switchCost = switchCost
        if initAction is not None:
            initActionDist = Multinomial([{'curAction': initAction}, ])
        self.initActionDist = initActionDist
    
    def isTerminal(self, s):
        return self.mdp.isTerminal(s['groundState'])
    
    def getNextStateDist(self, s, a) -> Multinomial:
        nsDist = self.mdp.getNextStateDist(s['groundState'], a)
        nsDist = Multinomial([{'groundState': ns, 'curAction': a} for ns in nsDist.support], logits=nsDist.logits)
        return nsDist
    
    def getReward(self, s, a, ns) -> float:
        r = self.mdp.getReward(s['groundState'], a, ns['groundState'])
        if s['curAction'] != a:
            r += self.switchCost
        return r

    def getActionDist(self, state) -> Multinomial:
        return self.mdp.getActionDist(state['groundState'])

    def getInitialStateDist(self) -> Multinomial:
        S0 = self.mdp.getInitialStateDist()
        if self.initActionDist is None:
            SA0 = Multinomial([])
            for s0 in S0.support:
                s0A0 = self.mdp.getActionDist(s0) #action distribution for this initial state
                s0A0 = Multinomial([{'groundState': s0, 'curAction': a} for a in s0A0.support], logits=s0A0.logits)
                SA0 = SA0 | s0A0
        else:
            SA0 = self.initActionDist
        S0 = Multinomial(support=[{'groundState': s} for s in S0.support], logits=S0.logits)
        return SA0 & S0