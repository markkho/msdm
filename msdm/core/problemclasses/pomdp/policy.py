import random
from collections import namedtuple
from abc import abstractmethod, ABC
from typing import TypeVar

from msdm.core.problemclasses.pomdp.pomdp import \
    State, Action, Observation, PartiallyObservableMDP
from msdm.core.distributions import Distribution
from msdm.core.algorithmclasses import Result

AgentState = TypeVar('AgentState')
Step = namedtuple("Step", "state agentstate action nextstate reward observation nextagentstate")

class POMDPPolicy(ABC):
    @abstractmethod
    def initial_agentstate(self) -> AgentState:
        pass

    @abstractmethod
    def action_dist(self, ag : AgentState) -> Distribution[Action]:
        pass

    @abstractmethod
    def next_agentstate(self, ag : AgentState, a : Action, o : Observation) -> AgentState:
        pass

    def run_on(self,
               pomdp: PartiallyObservableMDP,
               initial_state=None,
               initial_agentstate=None,
               max_steps=int(2 ** 30),
               rng=random):
        if initial_state is None:
            initial_state = pomdp.initial_state_dist().sample()
        if initial_agentstate is None:
            initial_agentstate = self.initial_agentstate()

        traj = []
        s = initial_state
        ag = initial_agentstate
        for t in range(max_steps):
            if pomdp.is_terminal(s):
                break
            a = self.action_dist(ag).sample(rng=rng)
            ns = pomdp.next_state_dist(s, a).sample(rng=rng)
            r = pomdp.reward(s, a, ns)
            o = pomdp.observation_dist(a, ns).sample(rng=rng)
            nag = self.next_agentstate(ag, a, o)
            traj.append(Step(s, ag, a, ns, r, o, nag))
            s = ns
            ag = nag
        traj.append(Step(s, ag, None, None, None, None, None))
        if traj:
            states, agentstates, actions, _, rewards, _, _ = zip(*traj)
        else:
            states = ()
            actions = ()
            rewards = ()
            agentstates = ()
        return traj
