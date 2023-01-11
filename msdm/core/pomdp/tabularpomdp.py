import logging
from collections import namedtuple
from abc import abstractmethod, ABC
from typing import Set, Sequence, Hashable, Mapping, TypeVar
import numpy as np

from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.pomdp.pomdp import PartiallyObservableMDP
from msdm.core.mdp import TabularMarkovDecisionProcess
from msdm.core.distributions import FiniteDistribution, DictDistribution

logger = logging.getLogger(__name__)

HashableObservation = TypeVar('HashableObservation', bound=Hashable)
Belief = namedtuple("Belief", "states probs")

class TabularPOMDP(TabularMarkovDecisionProcess, PartiallyObservableMDP):
    def as_matrices(self):
        return {
            'ss': self.state_list,
            'aa': self.action_list,
            'tf': self.transition_matrix,
            'rf': self.reward_matrix,
            'sarf': self.state_action_reward_matrix,
            's0': self.initial_state_vec,
            'rs': self.reachable_state_vec,
            'obs': self.observation_matrix
        }

    @cached_property
    def observation_list(self) -> Sequence[HashableObservation]:
        """
        List of observations. Note that ordering is only guaranteed to be
        consistent for a particular instance.
        """
        logger.info("Observation space unspecified; performing reachability analysis.")
        obs = set([])
        for a in self.action_list:
            for ns in self.state_list:
                obs.update([o for o, p in self.observation_dist(a, ns).items() if p > 0.])
        try:
            return sorted(obs)
        except TypeError: #unsortable representation
            pass
        return list(obs)

    @method_cache
    def _cached_observation_dist(self, a, ns) -> FiniteDistribution:
        '''
        We prefer using this cached version of observation_dist when possible.
        '''
        return self.observation_dist(a, ns)

    @cached_property
    def observation_index(self) -> Mapping[HashableObservation, int]:
        return {o: i for i, o in enumerate(self.observation_list)}

    @cached_property
    def observation_matrix(self) -> np.array:
        aa = self.action_list
        nss = self.state_list
        oo = self.observation_list
        ooi = self.observation_index
        obs = np.zeros((len(aa), len(nss), len(oo)))
        for ai, a in enumerate(self.action_list):
            for nsi, ns in enumerate(self.state_list):
                for o, p in self._cached_observation_dist(a, ns).items():
                    obs[ai, nsi, ooi[o]] = p
        return obs

    def state_estimator_vec(self, b: np.array, ai : int, oi : int) -> np.array:
        """
        Returns the posterior distribution over next states
        given an action, observation, and belief over previous states.

        This version is vectorized, and takes action/observations as indices.
        """

        dist = np.einsum('s,sn,n->n', b, self.transition_matrix[:, ai, :], self.observation_matrix[ai, :, oi])
        if dist.sum() == 0.0:
            return dist
        return dist/dist.sum()

    def predictive_observation_vec(self, b: np.array, ai : int) -> np.array:
        """
        Returns the predicted observation distribution for taking
        an action given a belief distribution.

        This version is vectorized, and takes actions as indices.
        """
        result = np.einsum('s,sn,no->o', b, self.transition_matrix[:, ai, :], self.observation_matrix[ai])
        assert np.isclose(result.sum(), 1)
        return result
