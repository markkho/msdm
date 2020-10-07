from typing import Mapping

from msdm.core.problemclasses.mdp.policy.policy import Policy
from msdm.core.distributions import DiscreteFactorTable, Distribution

class PartialPolicy(Policy):
    def __init__(self, policy_dict, default_actions=None):
        self._policydict = policy_dict
        self._defaults = default_actions

    def action_dist(self, s) -> Distribution:
        if (s not in self._policydict):
            if (self._defaults is None):
                raise Exception("No default action distribution set")
            raise NotImplementedError
        adist = self._policydict[s]
        assert isinstance(adist, dict)
        a, p = zip(*adist.items())
        return DiscreteFactorTable(support=a, probs=p)

    @property
    def policy_dict(self) -> Mapping:
        return self._policydict
