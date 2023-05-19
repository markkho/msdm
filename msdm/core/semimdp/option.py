import random
from abc import abstractmethod, abstractproperty, ABC
from typing import TypeVar, Generic, Sequence, Set, Tuple, Union, Callable, Any

from msdm.core.distributions import Distribution, DictDistribution
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.mdp import State, Action, MarkovDecisionProcess, Policy, \
    TabularMarkovDecisionProcess

from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.exceptions import AlgorithmException

class Option(ABC):
    name : str
    policy : Policy
    max_steps : int = 1000

    @abstractmethod
    def is_initial(self, s: State) -> bool:
        pass

    @abstractmethod
    def is_terminal(self, s: State) -> bool:
        pass

    def run_on(
            self,
            mdp: MarkovDecisionProcess,
            initial_state : State,
            rng : random.Random = random
        ):
        sub_mdp = augment(
            mdp=mdp,
            is_absorbing=lambda s : self.is_terminal(s),
        )
        result = self.policy.run_on(
            mdp=sub_mdp,
            initial_state=initial_state,
            max_steps=self.max_steps,
            rng=rng
        )
        if len(result) >= self.max_steps:
            raise AlgorithmException(
                f"{self} reached max steps ({self.max_steps}). " + \
                "It may be stuck in a loop or max_steps needs to be increased"
            )
        return result

class PlanToSubgoalOption(Option):
    _n_instances = 0
    def __init__(
        self,
        *,
        mdp : MarkovDecisionProcess,
        initial_states : Sequence[State],
        subgoals : Sequence[State],
        planner : Plans,
        include_mdp_absorbing_states : bool = False,
        name : str = None,
        max_steps : int = 1000,
        max_nonterminal_pseudoreward : float = float('inf'),
    ):
        if name is None:
            self.name = f"{self.__class__}_{self.__class__._n_instances}"
            self.__class__._n_instances += 1
        self.subgoals = subgoals
        self.initial_states = initial_states
        self.mdp = mdp
        self.name = name
        self.max_nonterminal_pseudoreward = max_nonterminal_pseudoreward
        self.include_mdp_absorbing_states = include_mdp_absorbing_states
        self.max_steps = max_steps
        self.planner = planner
    
    def is_initial(self, s: State) -> bool:
        return s in self.initial_states
    
    def is_terminal(self, s: State) -> bool:
        return s in self.subgoals
    
    @property
    def sub_task(self) -> MarkovDecisionProcess:
        def clipped_reward(s, a, ns):
            real_reward = self.mdp.reward(s, a, ns)
            if self.is_terminal(ns):
                return real_reward
            if real_reward > self.max_nonterminal_pseudoreward:
                return self.max_nonterminal_pseudoreward
            return real_reward
        def is_absorbing(s):
            if self.include_mdp_absorbing_states:
                is_absorbing = self.is_terminal(s) or self.mdp.is_absorbing(s)
            else:
                is_absorbing = self.is_terminal(s)
            return is_absorbing
        def initial_state_dist():
            return DictDistribution.uniform(self.initial_states)
        sub_mdp = augment(
            mdp = self.mdp,
            is_absorbing = is_absorbing,
            reward = clipped_reward,
            initial_state_dist = initial_state_dist,
        )
        return sub_mdp
    
    @property
    def planning_result(self) -> PlanningResult:
        return self.planner.plan_on(self.sub_task)
    
    @cached_property
    def policy(self) -> Policy:
        return self.planning_result.policy
    
    def __hash__(self) -> int:
        return hash(self.name)

def augment(
        mdp : MarkovDecisionProcess,
        initial_state_dist : Callable[[], Distribution] = None,
        actions : Callable[[State], Sequence[Action]] = None,
        next_state_dist : Callable[[State, Action], Distribution] = None,
        reward : Callable[[State, Action, State], float] = None,
        is_absorbing : Callable[[State], bool] = None,
        state_list : Sequence[State] = None,
        action_list : Sequence[Action] = None,
    ):
    """
    Returns an augmented `MarkovDecisionProcess` instance that safely overwrites core 
    methods/attributes of the given `MarkovDecisionProcess`, where the core methods are:
    `initial_state_dist`, `actions`, `next_state_dist`, `reward`, `is_absorbing`,
    `state_list`, and `action_list`
    """
    if state_list is not None or action_list is not None:
        assert isinstance(mdp, TabularMarkovDecisionProcess), \
            "state_list and action_list can only be set for TabularMarkovDecisionProcess"
    class AugmentedMDP(mdp.__class__):
        def __init__(self): pass
    if initial_state_dist is not None:
        AugmentedMDP.initial_state_dist = staticmethod(initial_state_dist)
    else:
        AugmentedMDP.initial_state_dist = mdp.initial_state_dist
    if actions is not None:
        AugmentedMDP.actions = staticmethod(actions)
    else:
        AugmentedMDP.actions = mdp.actions
    if next_state_dist is not None:
        AugmentedMDP.next_state_dist = staticmethod(next_state_dist)
    else:
        AugmentedMDP.next_state_dist = mdp.next_state_dist
    if reward is not None:
        AugmentedMDP.reward = staticmethod(reward)
    else:
        AugmentedMDP.reward = mdp.reward
    if is_absorbing is not None:
        AugmentedMDP.is_absorbing = staticmethod(is_absorbing)
    else:
        AugmentedMDP.is_absorbing = mdp.is_absorbing
    if (
        issubclass(AugmentedMDP, TabularMarkovDecisionProcess) and \
        isinstance(mdp, TabularMarkovDecisionProcess)
    ):
        if state_list is not None:
            AugmentedMDP.state_list = state_list
        else:
            AugmentedMDP.state_list = mdp.state_list
        if action_list is not None:
            AugmentedMDP.action_list = action_list
        else:
            AugmentedMDP.action_list = mdp.action_list
    return AugmentedMDP()