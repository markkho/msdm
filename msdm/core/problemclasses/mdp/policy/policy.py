from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping
from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess, State, Action
from msdm.core.distributions import Distribution
from msdm.core.algorithmclasses import Result
import random
import warnings

class Policy(ABC):
    @abstractmethod
    def action_dist(self, s: State) -> Distribution[Action]:
        pass

    def action(self, s: State) -> Action:
        return self.action_dist(s).sample()

    def run_on(
        self,
        mdp: MarkovDecisionProcess,
        initial_state=None,
        max_steps=int(2 ** 30),
        rng=random
    ):
        if initial_state is None:
            initial_state = mdp.initial_state_dist().sample()
        traj = []
        s = initial_state
        for t in range(max_steps):
            if mdp.is_terminal(s):
                break
            a = self.action_dist(s).sample(rng=rng)
            ns = mdp.next_state_dist(s, a).sample(rng=rng)
            r = mdp.reward(s, a, ns)
            traj.append(Step(
                timestep=t,
                state=s,
                action=a,
                next_state=ns,
                reward=r
            ))
            s = ns
        traj.append(Step(
            state=s,
        ))
        return SimulationResult(traj)

class Step(dict):
    timestep : int
    state : State
    action : Action
    next_state : State
    reward : float
    def __getattr__(self, attr):
        return self.get(attr, None)
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.items()])})"
    
class SimulationResult:
    def __init__(self, steps):
        self.steps = steps
    def __len__(self):
        return len(self.steps)
    def __getattr__(self, attr):
        return [step.get(attr, None) for step in self.steps]
    def __iter__(self):
        yield from self.rows
    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            assert len(key) == 2, "Too many dimensions"
            if isinstance(key[1], (tuple, list)):
                return [{k: step[k] for k in key[1]} for step in self.steps[key[0]]]
            else:
                return [step[key[1]] for step in self.steps[key[0]]]
        elif isinstance(key, (int, slice)):
            return self.steps[key]
        else:
            raise ValueError("Invalid key")
            
    # for backwards compatibility
    @property
    def state_traj(self):
        warnings.warn("`Result.state_traj` is deprecated, use `result['state']` or `result.state`")
        return tuple(self.state[:-1])
    @property
    def action_traj(self):
        warnings.warn("`Result.action_traj` is deprecated, use `result['action']` or `result.action`")
        return tuple(self.action[:-1])
    @property
    def reward_traj(self):
        warnings.warn("`Result.reward_traj` is deprecated, use `result['reward']` or `result.reward`")
        return tuple(self.reward[:-1])

@dataclass
class PolicyEvaluationResult:
    state_value : Mapping[State]
    action_value : Mapping[State, Mapping[Action]]
    initial_value : float
    state_occupancy : Mapping[State]
    n_simulations : float
