from abc import ABC, abstractmethod
from pyrlap_old.pyrlap2.core.problemclasses.mdp.mdp import MarkovDecisionProcess
from pyrlap_old.pyrlap2.core.distributions import Distribution

class Policy(ABC):
    @abstractmethod
    def getActionDist(self, s) -> Distribution:
        pass

    def runOn(self,
              mdp: MarkovDecisionProcess,
              initialState=None,
              maxSteps=20):
        if initialState is None:
            initialState = mdp.getInitialStateDist().sample()
        traj = []
        s = initialState
        for t in range(maxSteps):
            a = self.getActionDist(s).sample()
            ns = mdp.getNextStateDist(s, a).sample()
            r = mdp.getReward(s, a, ns)
            traj.append((s, a, ns, r))
            if mdp.isTerminal(s):
                break
            s = ns
        states, actions, nextstates, rewards = zip(*traj)
        return {
            'stateTraj': states,
            'actionTraj': actions,
            'rewardTraj': rewards
        }

