import random
from abc import abstractmethod, abstractproperty, ABC
from typing import TypeVar, Generic, Sequence, Set, Tuple, Union, Callable, Any

from msdm.core.distributions import Distribution, DictDistribution
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.mdp import State, Action, MarkovDecisionProcess, Policy, \
    TabularMarkovDecisionProcess

from msdm.algorithms import PolicyIteration

class Option(ABC):
    name : str

    @abstractmethod
    def is_initial(self, s: State) -> bool:
        pass

    @abstractmethod
    def is_terminal(self, s: State) -> bool:
        pass

    @abstractproperty
    def policy(self) -> Policy:
        pass

    @method_cache
    def sub_mdp(
        self,
        mdp : MarkovDecisionProcess,
        max_nonterminal_pseudoreward : float
    ) -> MarkovDecisionProcess:
        """
        The sub-MDP is the input MDP with an absorbing state function 
        set to the option's terminal state function. We also include a
        maximum pseudoreward for nonterminal states to prevent the
        option from staying in non-terminal states when used for planning.
        """
        # This is an inefficient but safe way to create a sub-MDP
        # that inherits all the noncached methods of the original MDP 
        # and replaces is_absorbing with the option's terminal state function.
        class SubMDP(mdp.__class__):
            def __init__(self_) -> None:
                self_.discount_rate = mdp.discount_rate
            def is_absorbing(self_, s: State) -> bool:
                return self.is_terminal(s)
            def next_state_dist(self_, s: State, a: Action) -> Distribution[State]:
                if self.is_terminal(s):
                    return DictDistribution({s: 1})
                return mdp.next_state_dist(s, a)
            def reward(self_, s: State, a: Action, ns: State) -> float:
                r = mdp.reward(s, a, ns)
                if not self.is_terminal(ns) and r >= max_nonterminal_pseudoreward:
                    return max_nonterminal_pseudoreward
                return r
            def initial_state_dist(self_) -> Distribution[State]:
                return mdp.initial_state_dist()
            def actions(self_, s: State) -> Sequence[Action]:
                return mdp.actions(s)
            @property
            def state_list(self_) -> Sequence[State]:
                assert isinstance(mdp, TabularMarkovDecisionProcess)
                return mdp.state_list
            @property
            def action_list(self_) -> Sequence[Action]:
                assert isinstance(mdp, TabularMarkovDecisionProcess)
                return mdp.action_list
        return SubMDP()
    
    def run_on(
            self,
            mdp: MarkovDecisionProcess,
            initial_state : State,
            max_steps : int = int(2**31),
            rng : random.Random = random
        ):
        sub_mdp = self.sub_mdp(
            mdp=mdp,
            max_nonterminal_pseudoreward=float('inf')
        )
        return self.policy.run_on(
            mdp=sub_mdp,
            initial_state=initial_state,
            max_steps=max_steps,
            rng=rng
        )

class SubgoalOption(Option):
    def __init__(
        self,
        mdp : MarkovDecisionProcess,
        subgoals : Sequence[State],
        name : str,
        max_nonterminal_pseudoreward : float
    ):
        self.subgoals = subgoals
        self.mdp = mdp
        self.name = name
        self.max_nonterminal_pseudoreward = max_nonterminal_pseudoreward
    
    def is_initial(self, s: State) -> bool:
        return True
    
    def is_terminal(self, s: State) -> bool:
        return s in self.subgoals
    
    @cached_property
    def policy(self) -> Policy:
        sub_mdp = self.sub_mdp(
            self.mdp,
            max_nonterminal_pseudoreward=self.max_nonterminal_pseudoreward
        )
        pi_res = PolicyIteration().plan_on(sub_mdp)
        return pi_res.policy
    
    def __hash__(self) -> int:
        return hash(self.name)
