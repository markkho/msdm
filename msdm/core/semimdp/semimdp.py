
import dataclasses
from collections import defaultdict
import random
from typing import TypeVar, Generic, Sequence, Set, Tuple, Union, Callable, Any

from msdm.core.distributions import Distribution, DictDistribution
from msdm.core.distributions.utils import obj_seed
from msdm.core.mdp import State, Action, MarkovDecisionProcess
from msdm.core.mdp.policy import SimulationResult

from msdm.core.semimdp.option import Option

@dataclasses.dataclass()
class SemiMarkovDecisionProcess:
    mdp : MarkovDecisionProcess
    options : Sequence[Option]
    n_option_simulations : int
    include_mdp_actions : bool = False
    seed : int = None

    def initial_state_dist(self) -> Distribution:
        return self.mdp.initial_state_dist()

    def actions(self, s: State) -> Sequence[Union[Action, Option]]:
        available_options = [o for o in self.options if o.is_initial(s)]
        if self.include_mdp_actions:
            return self.mdp.actions(s) + available_options
        else:
            return available_options
        
    def expected_cumulative_reward(
        self,
        s: State,
        a: Union[Action, Option],
    ) -> float:
        return self.next_state_transit_time_reward_dist(s, a).expectation(
            lambda ns_t_r: ns_t_r[2]
        )

    def next_state_transit_time_dist(
        self,
        s: State,
        a: Union[Action, Option],
    ) -> Distribution:
        return self.next_state_transit_time_reward_dist(s, a).marginalize(
            lambda ns_t_r: (ns_t_r[0], ns_t_r[1])
        )
    
    def next_state_dist(
        self,
        s: State,
        a: Union[Action, Option],
    ) -> Distribution:
        return self.next_state_transit_time_reward_dist(s, a).marginalize(
            lambda ns_t_r: ns_t_r[0]
        )
    
    def next_state_transit_time_reward_dist(
        self,
        s: State,
        a: Union[Action, Option],
    ) -> DictDistribution:
        if a in self.mdp.actions(s):
            ns_dist : DictDistribution = self.mdp.next_state_dist(s, a)
            return ns_dist.marginalize(lambda ns: (ns, 1, self.mdp.reward(s, a, ns)))
        elif isinstance(a, Option):
            simulations = self.run_simulations(s, a)
            counts = defaultdict(int)
            for sim in simulations:
                ns = sim.state[0]
                discount = 1
                cum_reward = 0
                t = 0
                for ns_, r in zip(sim.next_state, sim.reward):
                    cum_reward += r*discount
                    discount = discount*self.mdp.discount_rate
                    if ns_ is not None:
                        ns = ns_
                        t += 1
                counts[(ns, t, cum_reward)] += 1
            nstr_dist = \
                DictDistribution({
                    ns_t_r: c/self.n_option_simulations for ns_t_r, c in counts.items()
                })
            return nstr_dist
        else:
            raise ValueError(f"{a} is not a ground action or option")

    def run_simulations(
        self,
        s: State,
        a: Option,
    ) -> Sequence[SimulationResult]:
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        seed = obj_seed((s, a, self.seed))
        rng = random.Random(seed)
        simulations = []
        for _ in range(self.n_option_simulations):
            simulation = a.run_on(self.mdp, initial_state=s, rng=rng)
            simulations.append(simulation)
        return simulations
