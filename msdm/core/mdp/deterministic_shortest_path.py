from abc import abstractmethod
from msdm.core.mdp.mdp import MarkovDecisionProcess
from msdm.core.distributions import Distribution, DeterministicDistribution

class DeterministicShortestPathProblem(MarkovDecisionProcess):
    """
    A DeterministicShortestPathProblem is a special case of an MDP, with some small differences:
    - A deterministic transition function.
    - A deterministic initial state.
    - A negative reward function.
    """

    def next_state_dist(self, s, a) -> Distribution:
        return DeterministicDistribution(self.next_state(s, a))

    @abstractmethod
    def next_state(self, s, a):
        pass

    def initial_state_dist(self) -> Distribution:
        return DeterministicDistribution(self.initial_state())

    @abstractmethod
    def initial_state(self):
        pass
