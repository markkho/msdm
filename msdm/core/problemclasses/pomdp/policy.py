from abc import abstractmethod, ABC
from typing import TypeVar

from msdm.core.problemclasses.pomdp.pomdp import \
    State, Action, Observation
from msdm.core.distributions import Distribution

AgentState = TypeVar('AgentState')

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
