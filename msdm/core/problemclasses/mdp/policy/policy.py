from abc import ABC, abstractmethod
from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess
from msdm.core.distributions import Distribution
from msdm.core.algorithmclasses import Result

class Policy(ABC):
    @abstractmethod
    def action_dist(self, s) -> Distribution:
        pass

    def run_on(self,
               mdp: MarkovDecisionProcess,
               initial_state=None,
               max_steps=int(2 ** 30)):
        if initial_state is None:
            initial_state = mdp.initial_state_dist().sample()
        traj = []
        s = initial_state
        for t in range(max_steps):
            a = self.action_dist(s).sample()
            ns = mdp.next_state_dist(s, a).sample()
            r = mdp.reward(s, a, ns)
            traj.append((s, a, ns, r))
            if mdp.is_terminal(ns):
                break
            s = ns
        states, actions, nextstates, rewards = zip(*traj)
        return Result(**{
            'state_traj': states,
            'action_traj': actions,
            'reward_traj': rewards
        })

