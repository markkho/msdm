from msdm.core.pomdp.policy import POMDPPolicy, AgentState
from msdm.core.pomdp.pomdp import Action, Observation
from msdm.core.pomdp.tabularpomdp import TabularPOMDP
from msdm.core.distributions import Distribution, DictDistribution

class FiniteStateController(POMDPPolicy):
    def __init__(self, pomdp: TabularPOMDP, action_strategy, observation_strategy, *, initial_state=0):
        nstates = len(action_strategy)
        assert nstates == observation_strategy.shape[0]
        assert len(observation_strategy.shape) in (2, 3)
        if len(observation_strategy.shape) == 2:
            assert observation_strategy.shape == (
                nstates, len(pomdp.action_list), len(pomdp.observation_list)
            )
        else:
            assert observation_strategy.shape == (
                nstates, len(pomdp.observation_list)
            )

        self.action_strategy = action_strategy
        self.observation_strategy = observation_strategy
        self.initial_state = initial_state
        self.pomdp = pomdp

    def initial_agentstate(self) -> AgentState:
        return self.initial_state

    def action_dist(self, ag : AgentState) -> Distribution[Action]:
        action = self.action_strategy[ag]
        action_idx = self.pomdp.action_list.index(action)
        return DictDistribution.deterministic(action_idx)

    def next_agentstate(self, ag : AgentState, a : Action, o : Observation) -> AgentState:
        oi = self.pomdp.observation_index[o]
        ai = self.pomdp.action_list.index(a)

        if len(self.observation_strategy.shape) == 3:
            return self.observation_strategy[ag, ai, oi]
        else:
            return self.observation_strategy[ag, oi]

class StochasticFiniteStateController(POMDPPolicy):
    def __init__(self, pomdp: TabularPOMDP, action_strategy, observation_strategy, initial_state_dist):
        ncontroller = action_strategy.shape[0]
        nactions, nstates, nobs = pomdp.observation_matrix.shape

        assert action_strategy.shape == (ncontroller, nactions)
        assert observation_strategy.shape == (ncontroller, nactions, nobs, ncontroller)
        assert initial_state_dist.shape == (ncontroller,)

        self.action_strategy = action_strategy
        self.observation_strategy = observation_strategy
        self.initial_state_dist = initial_state_dist
        self.pomdp = pomdp

    def initial_agentstate(self) -> AgentState:
        return self.initial_state_dist

    def action_dist(self, ag : AgentState) -> Distribution[Action]:
        action_dist = ag @ self.action_strategy
        return DictDistribution({
            a: action_dist[ai] for ai, a in enumerate(self.pomdp.action_list)
        })

    def next_agentstate(self, ag : AgentState, a : Action, o : Observation) -> AgentState:
        oi = self.pomdp.observation_index[o]
        ai = self.pomdp.action_list.index(a)
        return ag @ self.observation_strategy[:, ai, oi]
