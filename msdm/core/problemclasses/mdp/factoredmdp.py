from msdm.core.problemclasses.mdp import MarkovDecisionProcess
from msdm.core.distributions.jointprobabilitytable import \
    Assignment, JointProbabilityTable

class FactoredMDP(MarkovDecisionProcess):
    def __init__(
        self,
        initial_state_factors,
        next_state_factors,
        reward_factors,
        next_state_variable_substring="next_",
        reward_variable_substring="_reward",
    ):
        self._initial_state_factors = initial_state_factors
        self._next_state_factors = next_state_factors
        self._reward_factors = reward_factors
        self.next_state_variable_substring = next_state_variable_substring
        self.reward_variable_substring = reward_variable_substring

    def initial_state_dist(self):
        return JointProbabilityTable.\
            null_table().\
            join(*self._initial_state_factors)

    def next_state_dist(self, s, a):
        dist = JointProbabilityTable.deterministic(s+a)
        dist = dist.join(*self._next_state_factors)
        ns_dist = \
            dist.groupby(lambda c: self.next_state_variable_substring in c)
        ns_dist = \
            ns_dist.rename_columns(lambda c: c.replace(self.next_state_variable_substring, ""))
        ns_dist = ns_dist.normalize()
        return ns_dist

    def reward(self, s, a, ns):
        """
        Following definitions in [Koller & Milch (2001)
        Multi-agent influence diagrams for representing
        and solving games], total reward is an additive combination
        of reward variables, which take on real values and
        are determinstic given their parent variables.
        """
        rdist = JointProbabilityTable.deterministic(s+a)
        rdist = rdist.join(*self._reward_factors)
        assert len(rdist) == 1
        rdist = rdist.groupby(lambda c: self.reward_variable_substring in c)
        return sum(rdist.first()[0].values())
