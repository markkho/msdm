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

    @classmethod
    def from_mdp(cls, mdp: MarkovDecisionProcess) -> 'DeterministicShortestPathProblem':
        """
        Safely converts an MDP into a DeterministicShortestPathProblem.
        """
        if isinstance(mdp, DeterministicShortestPathProblem):
            return mdp
        class DeterministicShortestPathProblemFromMDP(DeterministicShortestPathProblem):
            def __init__(self): pass
            def initial_state(self):
                initial_state = mdp.initial_state_dist().support
                assert len(initial_state) == 1, "MDP has non-deterministic initial state"
                return initial_state[0]
            def next_state(self, s, a):
                next_state = mdp.next_state_dist(s, a).support
                assert len(next_state) == 1, "MDP has non-deterministic transition function"
                return next_state[0]
        DeterministicShortestPathProblemFromMDP.actions = staticmethod(mdp.actions)
        DeterministicShortestPathProblemFromMDP.reward = staticmethod(mdp.reward)
        DeterministicShortestPathProblemFromMDP.is_absorbing = staticmethod(mdp.is_absorbing)
        return DeterministicShortestPathProblemFromMDP()
