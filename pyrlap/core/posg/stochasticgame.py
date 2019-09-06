from typing import Union, Iterable, Mapping

from pyrlap.core.mdp import MDP
from pyrlap.core.base import State, Action, Observation, Probability


class StochasticGame(MDP):
    """
    A stochastic game is a modified MDP where actions
    and rewards are tuples of length <number of agents>.
    """
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def reward(self,
               s=None,
               a : Union([Iterable, None]) = None,
               ns=None) -> Iterable[float]:
        raise NotImplementedError

    def transition(self, s, a : Iterable[Action]) -> State:
        raise NotImplementedError

    def transition_dist(self, s, a: Iterable[Action]) -> Mapping[State, Probability]:
        raise NotImplementedError