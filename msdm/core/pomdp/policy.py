import random
from collections import namedtuple
from abc import abstractmethod, ABC
from typing import TypeVar

from msdm.core.pomdp.pomdp import \
    State, Action, Observation, PartiallyObservableMDP
from msdm.core.pomdp.tabularpomdp import TabularPOMDP, Belief
from msdm.core.distributions import Distribution, DictDistribution
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
            if pomdp.is_absorbing(s):
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

class ValueBasedTabularPOMDPPolicy(POMDPPolicy):
    """
    POMDP policy that selects actions based on a
    representation of action values at a belief state.
    """
    def __init__(self, pomdp : TabularPOMDP):
        self.pomdp = pomdp

    @abstractmethod
    def action_value(self, b : Belief, a : Action):
        pass

    def initial_agentstate(self):
        return Belief(tuple(self.pomdp.state_list), tuple(self.pomdp.initial_state_vec))

    def action_dist(self, ag : Belief):
        av = {a: self.action_value(ag, a) for a in self.pomdp.action_list}
        maxv = max(av.values())
        return DictDistribution.uniform([a for a, v in av.items() if v == maxv])

    def next_agentstate(self, ag, a, o):
        s_dist = DictDistribution(zip(*ag))
        ns_dist = self.pomdp.state_estimator(s_dist, a, o)
        ss = tuple(self.pomdp.state_list)
        return Belief(ss, tuple([ns_dist.prob(ns) for ns in ss]))
