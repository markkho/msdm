from abc import ABC, abstractmethod
from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess
from msdm.core.distributions import Distribution

class Policy(ABC):
    @abstractmethod
    def action_dist(self, s) -> Distribution:
        pass

    def run_on(self,
               mdp: MarkovDecisionProcess,
               initialState=None,
               maxSteps=int(2**30)):
        if initialState is None:
            initialState = mdp.initial_state_dist().sample()
        traj = []
        s = initialState
        for t in range(maxSteps):
            a = self.action_dist(s).sample()
            ns = mdp.next_state_dist(s, a).sample()
            r = mdp.reward(s, a, ns)
            traj.append((s, a, ns, r))
            if mdp.is_terminal(ns):
                break
            s = ns
        states, actions, nextstates, rewards = zip(*traj)
        return {
            'stateTraj': states,
            'actionTraj': actions,
            'rewardTraj': rewards
        }

