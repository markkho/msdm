from abc import abstractmethod
from collections.abc import Hashable, Mapping, Iterable

from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.distributions import Distribution
from msdm.core.utils.hashdictionary import HashDictionary, DefaultHashDictionary


class MarkovDecisionProcess(ProblemClass):
    @abstractmethod
    def next_state_dist(self, s, a) -> Distribution:
        pass

    @abstractmethod
    def reward(self, s, a, ns) -> float:
        pass

    @abstractmethod
    def actions(self, s) -> Iterable:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def is_terminal(self, s):
        pass

    def hash_state(self, s) -> Hashable:
        raise NotImplementedError

    def hash_action(self, a) -> Hashable:
        raise NotImplementedError

    def _variable_map(self,
                      hashable,
                      hash_function=None,
                      default_value=None,
                      initialize_defaults=True):
        """Generic function for mapping variables"""
        if default_value is not None:
            return DefaultHashDictionary(
                default_value=default_value,
                initialize_defaults=initialize_defaults,
                hash_function=hash_function
            )
        if hashable:
            return {}
        else:
            return HashDictionary(hash_function=hash_function)

    def state_map(self, default_value=None) -> Mapping:
        """Creates a dictionary-like object where keys are states."""
        s0 = self.initial_state_dist().sample()
        if isinstance(s0, Hashable):
            return self._variable_map(
                hashable=True,
                default_value=default_value
            )
        return self._variable_map(
            hashable=False,
            hash_function=self.hash_state,
            default_value=default_value,
        )

    def action_map(self, default_value=None) -> Mapping:
        """Creates a dictionary-like object where keys are actions."""
        s0 = self.initial_state_dist().sample()
        a = next(iter(self.actions(s0)))
        if isinstance(a, Hashable):
            return self._variable_map(
                hashable=True,
                default_value=default_value
            )
        return self._variable_map(
            hashable=False,
            hash_function=self.hash_action,
            default_value=default_value,
        )

    def state_action_map(self, default_value=None):
        def make_action_map(s):
            return self.action_map(default_value=default_value)
        return self.state_map(default_value=make_action_map)
