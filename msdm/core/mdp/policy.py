from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import random
import warnings
from typing import Callable, Collection, Mapping, Sequence

import numpy as np

from msdm.core.mdp.mdp import \
    MarkovDecisionProcess, State, Action
from msdm.core.mdp.tables import StateTable, StateActionTable
from msdm.core.distributions import Distribution

class Policy(ABC):
    @abstractmethod
    def action_dist(self, s: State) -> Distribution[Action]:
        pass

    def action(self, s: State) -> Action:
        return self.action_dist(s).sample()
    
    def to_tabular(
        self,
        state_list : Sequence[State],
        action_list : Sequence[Action]
    ):
        # import here to avoid circular dependency
        from msdm.core.mdp.tabularpolicy import TabularPolicy
        policy_matrix = np.zeros((len(state_list), len(action_list)))
        action_index = {a: ai for ai, a in enumerate(action_list)}
        for si, s in enumerate(state_list):
            for a, prob in self.action_dist(s).items():
                ai = action_index[a]
                policy_matrix[si, ai] = prob
        return TabularPolicy.from_state_action_lists(
            state_list=state_list,
            action_list=action_list,
            data=policy_matrix
        ) 

    def run_on(
        self,
        mdp: MarkovDecisionProcess,
        initial_state=None,
        max_steps=int(2 ** 30),
        rng=random
    ):
        if initial_state is None:
            initial_state = mdp.initial_state_dist().sample(rng=rng)
        traj = []
        s = initial_state
        for t in range(max_steps):
            if mdp.is_absorbing(s):
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
    
    def evaluate_on(
        self,
        mdp: MarkovDecisionProcess,
        n_simulations=100,
        max_steps=int(2 **30),
        rng=random
    ):
        state_value_samples = defaultdict(list)
        state_samples = defaultdict(float)
        action_value_samples = defaultdict(lambda : defaultdict(list))
        initial_values = []
        for _ in range(n_simulations):
            res = self.run_on(mdp, rng=rng, max_steps=max_steps)
            rets = Policy.calc_returns(res.reward, mdp.discount_rate)
            initial_values.append(rets[0])
            for ret, s, a in zip(rets, res.state, res.action):
                state_value_samples[s].append(ret)
                action_value_samples[s][a].append(ret)
        state_value = {}
        action_value = {}
        for s, state_samps in state_value_samples.items():
            state_samples[s] += len(state_samps)/n_simulations
            state_value[s] = np.mean(state_samps)
            action_value[s] = {}
            for a, action_samps in action_value_samples[s].items():
                action_value[s][a] = np.mean(action_samps)
        return PolicyEvaluationResult(
            state_value=StateTable.from_dict(state_value),
            action_value=StateActionTable.from_dict(action_value, default_value=float('-inf')),
            initial_value=np.mean(initial_values),
            state_occupancy=StateTable.from_dict(state_samples),
            n_simulations=n_simulations,
        )

    @staticmethod
    def calc_returns(rewards, discount_rate):
        rs = np.array(rewards)
        times = np.arange(len(rs))
        rel_times = times - times[:, np.newaxis]
        rel_times = rel_times
        discounts = np.triu(np.power(discount_rate, rel_times))
        rets = discounts@rs
        return list(rets)

class FunctionalPolicy(Policy):
    def __init__(self, function: Callable[[State], Mapping[Action, float]]):
        self._function = function
    
    def action_dist(self, s: State):
        return self._function(s)

class Step(dict):
    def __getattr__(self, attr):
        return self.get(attr, None)
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.items()])})"
    
class SimulationResult:
    def __init__(self, steps):
        self.steps = steps
    def __len__(self):
        return len(self.steps)
    @property
    def reward(self):
        return [s.get('reward', 0) for s in self.steps]
    @property
    def action(self):
        return [s.get('action', None) for s in self.steps]
    @property
    def state(self):
        return [s.get('state', None) for s in self.steps]
    @property
    def next_state(self):
        return [s.get('next_state', None) for s in self.steps]
    def __eq__(self, other : "SimulationResult"):
        return all(s == o for s, o in zip(self.steps, other.steps))
    def __iter__(self):
        yield from self.steps
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

    def _repr_html_(self):
        import pandas as pd
        return pd.DataFrame([dict(s) for s in self.steps]).to_html()
            
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
    state_value : Mapping[State, float]
    action_value : Mapping[State, Mapping[Action, float]]
    initial_value : float
    state_occupancy : Mapping[State, float]
    n_simulations : float
