from abc import abstractmethod
from msdm.core.problemclasses.mdp import MarkovDecisionProcess
from msdm.core.distributions import Distribution, Multinomial, DictDistribution

class DeterministicShortestPathProblem(MarkovDecisionProcess):
    """
    A DeterministicShortestPathProblem is a special case of an MDP, with some small differences:
    - A deterministic transition function.
    - A deterministic initial state.
    - A negative reward function.
    """

    def next_state_dist(self, s, a) -> Distribution:
        return DictDistribution({self.next_state(s, a): 1})

    @abstractmethod
    def next_state(self, s, a):
        pass

    def initial_state_dist(self) -> Distribution:
        return DictDistribution({self.initial_state(): 1})

    @abstractmethod
    def initial_state(self):
        pass
