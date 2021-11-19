from abc import ABC, abstractmethod
from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess, State, Action
from msdm.core.distributions import Distribution
from msdm.core.algorithmclasses import Result
import random

class Policy(ABC):
    @abstractmethod
    def action_dist(self, s: State) -> Distribution[Action]:
        pass

    def action(self, s: State) -> Action:
        return self.action_dist(s).sample()

    def run_on(self,
               mdp: MarkovDecisionProcess,
               initial_state=None,
               max_steps=int(2 ** 30),
               rng=random):
        if initial_state is None:
            initial_state = mdp.initial_state_dist().sample()
        traj = []
        s = initial_state
        for t in range(max_steps):
            if mdp.is_terminal(s):
                break
            a = self.action_dist(s).sample(rng=rng)
            ns = mdp.next_state_dist(s, a).sample(rng=rng)
            r = mdp.reward(s, a, ns)
            traj.append((s, a, ns, r))
            s = ns
        if traj:
            states, actions, _, rewards = zip(*traj)
        else:
            states = ()
            actions = ()
            rewards = ()
        return Result(**{
            'state_traj': states,
            'action_traj': actions,
            'reward_traj': rewards
        })
