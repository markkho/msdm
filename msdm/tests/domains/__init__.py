from typing import Dict, Sequence
from itertools import product

from collections import defaultdict
from msdm.domains.gridworld.mdp import GridWorld

from msdm.core.mdp import TabularMarkovDecisionProcess, DeterministicShortestPathProblem
from msdm.core.utils.funcutils import cached_property
from msdm.core.distributions import \
    Distribution, DictDistribution,\
    DeterministicDistribution, UniformDistribution
from msdm.tests.domains.russell_norvig import make_russell_norvig_grid
from msdm.tests.domains.slipperymaze import SlipperyMaze

class TestDomain:
    def optimal_policy(self):
        raise NotImplementedError
    def optimal_state_value(self):
        raise NotImplementedError

class GNTFig6_6(TabularMarkovDecisionProcess):
    T = [
        (((1, 2), 5), ((7,), 19), ((3, 9), 12)),
        (((4, 5), 4), ((11, 13), 4), ((6,), 2)),
        (((11,), 20), ((6,), 4)),
        (((6,), 8), ((8, 9), 5)),
        (((10,), 5), ((11, 12), 3)),
        (((11, 12), 4),),
        (((13, 14, 15), 5),),
        (((13,), 20), ((14, 15), 15)),
        (((14, 15), 6), ((9,), 4)),
        (((14, 15), 9),),
        (((5, 12), 7),),
        (((12,), 10), ((13, 14), 6)),
        (),
        (((14, 16), 35),),
        (((15, 16), 25),),
        (),
        (),
    ]

    '''
    Acyclic MDP from Ghallab, Nau, Traverso Figure 6.6
    '''
    def initial_state_dist(self) -> Distribution:
        return DeterministicDistribution(0)

    def is_absorbing(self, s):
        return s in (12, 15, 16)

    def actions(self, s) -> Sequence:
        return [0, 1, 2]
        # dests = GNTFig6_6.T[s]
        # return [a for a in range(len(dests)) if dests[a][0]]

    def next_state_dist(self, s, a):
        if a < len(GNTFig6_6.T[s]):
            ns = GNTFig6_6.T[s][a][0]
        else:
            ns = [s]
        return UniformDistribution(ns)

    def reward(self, s, a, ns) -> float:
        if a < len(GNTFig6_6.T[s]):
            return -GNTFig6_6.T[s][a][1]
        return -100 # HACK

class DeterministicCounter(DeterministicShortestPathProblem, TabularMarkovDecisionProcess, TestDomain):
    '''
    MDP where actions are increment/decrement and goal is to reach some count.
    '''
    def __init__(self, goal, *, initial_state=0, discount_rate=1.0):
        self._initial_state = initial_state
        self.goal = goal
        self.discount_rate = discount_rate

    def initial_state(self):
        return self._initial_state

    def actions(self, s):
        return [1, -1]

    def next_state(self, s, a):
        ns = s + a
        if ns < 0 or self.goal < ns:
            return s
        return ns

    def reward(self, s, a, ns):
        return -1

    def is_absorbing(self, s):
        return s == self.goal
    
    def optimal_policy(self):
        policy = {}
        for s in self.state_list:
            if self.is_absorbing(s):
                policy[s] = DictDistribution({1: .5, -1: .5})
            else:
                policy[s] = DictDistribution({1: 1})
        return policy
    def optimal_state_value(self):
        value = {}
        for s in self.state_list:
            steps_away = self.goal - s
            discount = 1
            value[s] = 0
            for _ in range(steps_away):
                value[s] += -1*discount
                discount *= self.discount_rate
        return value

class DeterministicUnreachableCounter(DeterministicCounter):
    @property
    def state_list(self):
        return (float('-inf'), -1) + tuple(range(self.goal + 1))
    def optimal_policy(self):
        policy = {}
        for s in self.state_list:
            if self.is_absorbing(s) or s == float('-inf'):
                policy[s] = DictDistribution({1: .5, -1: .5})
            else:
                policy[s] = DictDistribution({1: 1})
        return policy
    def optimal_state_value(self):
        value = {}
        for s in self.state_list:
            if s == float('-inf'):
                if self.discount_rate < 1.0:
                    value[s] = -1/(1 - self.discount_rate)
                else:
                    value[s] = 0
                continue
            
            steps_left = self.goal - s
            v = 0
            for i in range(steps_left):
                v += -1*(self.discount_rate**i)
            value[s] = v
        return value
    def optimal_state_gain(self):
        value = {s: 0 for s in self.state_list}
        if self.discount_rate == 1.0:
            value[float('-inf')] = -1
        return value

class LineWorld(TabularMarkovDecisionProcess):
    def __init__(self, line="s..#...#s.g", discount_rate=1.0):
        self.line = line
        self.discount_rate = discount_rate
    @cached_property
    def state_list(self):
        return tuple(range(len(self.line)))
    def next_state_dist(self, s, a):
        ns = s + a
        if not (0 <= ns < len(self.line)) or self.line[ns] == '#':
            ns = s
        return DictDistribution({ns: 1})
    def is_absorbing(self, s) -> bool:
        return self.line[s] == 'g'
    def actions(self, s):
        return (-1, 0, 1)
    def initial_state_dist(self):
        initial_states = [i for i, c in enumerate(self.line) if c == 's']
        return DictDistribution.uniform(initial_states)
    def reward(self, s, a, ns) -> float:
        return -1

class GeometricCounter(TabularMarkovDecisionProcess, TestDomain):
    '''
    MDP where actions are to draw from a Bernoulli or wait.
    Goal is to get a 1 from the Bernoulli, which has probability `p`.
    '''
    def __init__(self, *, p=1/2, discount_rate=1.0):
        self.p = p
        self.discount_rate = discount_rate

    def initial_state_dist(self):
        return DictDistribution({0: 1})

    def actions(self, s):
        return ['flip', 'wait']

    def next_state_dist(self, s, a):
        if a == 'wait':
            return DictDistribution({s: 1})
        elif a == 'flip':
            return DictDistribution({0: 1 - self.p, 1: self.p})

    def reward(self, s, a, ns):
        return -1

    def is_absorbing(self, s):
        return s == 1
    
    def optimal_policy(self):
        return {
            0: DictDistribution({'flip': 1}),
            1: DictDistribution({'wait': .5, 'flip': .5}),
        }
    def optimal_state_value(self):
        return {
            0: -1/(1 - (1 -self.p)*self.discount_rate),
            1: 0
        }

class VaryingActionNumber(DeterministicShortestPathProblem, TabularMarkovDecisionProcess, TestDomain):
    '''
    Counting MDP where actions at every state vary. Used to test handling of MDPs with
    varying numbers of states.
    '''
    def __init__(self, discount_rate=1.0):
        self.discount_rate = discount_rate
    def initial_state(self):
        return 0

    def actions(self, s):
        return {
            0: [+1],
            1: [-1, +1],
            2: [-1],
        }[s]

    def next_state(self, s, a):
        # Intentionally coding like this to ensure that
        # invalid access of this function results in a KeyError
        return {
            (0, +1): 1,
            (1, -1): 0,
            (1, +1): 2,
            (2, -1): 1,
        }[s, a]

    def reward(self, s, a, ns):
        return -1

    def is_absorbing(self, s):
        return s == 2

    def optimal_policy(self):
        return {
            0: DictDistribution({
                1: 1
            }),
            1: DictDistribution({
                1: 1
            }),
            2: DictDistribution({
                -1: 1
            }),
        }

    def optimal_state_value(self):
        return {
            0: -2,
            1: -1,
            2: 0
        }

class DeadEndBandit(TabularMarkovDecisionProcess, TestDomain):
    """
    An example of an MDP with deadends.
    From Ghallab, Nau & Traverso Figure 6.1
    """
    @property
    def state_list(self):
        return tuple([''.join(s) for s in product("abc", "abc", "abc")])
    def initial_state_dist(self):
        return DictDistribution({"abc": 1})
    def actions(self, s):
        if len(set(s)) == 3 or self.is_absorbing(s):
            return ('left', 'right', 'mid')
        if s[0] != s[1] == s[2]:
            return ('left',)
        if s[0] == s[1] != s[2]:
            return ('right',)
        if s[0] == s[2] != s[1]:
            return ('mid',)
        if s[0] == s[1] == s[2]:
            return ()
    def is_absorbing(self, s):
        return s in ['bbb', 'ccc']
    def reward(self, s, a, ns):
        return -1
    def next_state_dist(self, s, a):
        i = {'left': 0, 'mid': 1, 'right': 2}[a]
        return DictDistribution({s[:i]+letter+s[i+1:]: 1/3 for letter in 'abc'})
        
    def optimal_policy(self):
        """
        The optimal policy is to pull the arm of 'a' whenever there is a single
        'a'. Otherwise, its an absorbing/dead end/single action state.
        """
        policy = {}
        for s in self.state_list:
            actions = self.actions(s)
            if len(actions) == 0:
                # dead ends are handled by turning them into a uniform distribution
                # since they are all equally -inf
                policy[s] = DictDistribution({a: 1/len(self.action_list) for a in self.action_list})
            elif len(actions) == 1:
                policy[s] = DictDistribution({actions[0]: 1})
            elif s in ['bbb', 'ccc']:
                policy[s] = DictDistribution({a: 1/len(actions) for a in actions})
            else:
                optimal_action = {0:'left', 1:'mid', 2:'right'}[s.index('a')]
                policy[s] = DictDistribution({optimal_action: 1})
        return policy

    def optimal_state_value(self):
        value = {}
        for s in self.state_list:
            letter_counts = {
                'a': sum([c == 'a' for c in s]),
                'b': sum([c == 'b' for c in s]),
                'c': sum([c == 'c' for c in s]),
            }
            if letter_counts['a'] >= 2:
                value[s] = float('-inf')
            elif letter_counts['b'] == letter_counts['c'] == letter_counts['a']:
                value[s] = -4.5
            elif letter_counts['b'] == 2:
                value[s] = -3
            elif letter_counts['c'] == 2:
                value[s] = -3
            elif letter_counts['b'] == 3:
                value[s] = 0
            elif letter_counts['c'] == 3:
                value[s] = 0
            else:
                raise
        return value

class TransitionRewardDictMDP(TabularMarkovDecisionProcess):
    _transition_rewards : dict
    def actions(self, s):
        actions = tuple(self._transition_rewards[s].keys())
        return actions
    def next_state_dist(self, s, a):
        ns_prob_r = self._transition_rewards[s][a]
        ns_prob = {ns: p for ns, (p, r) in ns_prob_r.items()}
        return DictDistribution(ns_prob)
    def reward(self, s, a, ns):
        ns_prob_r = self._transition_rewards[s][a]
        ns_r = {ns: r for ns, (p, r) in ns_prob_r.items()}
        return ns_r[ns]

class PositiveRewardCycle(TransitionRewardDictMDP, TestDomain):
    _transition_rewards = {
        'a': {
            'b': {'b': (1, -1)},
            'c': {'c': (1, -10)}
        },
        'b': {
            'a': {'a': (1, -10)},
            'c': {'c': (1, -1)}
        },
        'c': {
            'a': {'a': (1, 3)},
            'b': {'b': (1, -3)}
        }
    }
    def __init__(self, discount_rate=.95):
        self.discount_rate = discount_rate
    def initial_state_dist(self):
        return DictDistribution({'a': 1/3, 'b': 1/3, 'c': 1/3})
    def is_absorbing(self, s):
        return False
    def optimal_policy(self):
        return {
            'a': DictDistribution({'b': 1}),
            'b': DictDistribution({'c': 1}),
            'c': DictDistribution({'a': 1}),
        }
    def optimal_state_value(self):
        value_from_c = 0
        discount = 1
        for i in range(1000):
            if i % 3 == 0:
                value_from_c += 3*discount
            else:
                value_from_c += -1*discount
            discount *= self.discount_rate
        return {
            'a': -1 + -self.discount_rate + value_from_c*(self.discount_rate**2),
            'b': -1 + value_from_c*(self.discount_rate),
            'c': value_from_c
        }

class Puterman_Example_9_1_1(TransitionRewardDictMDP,TestDomain):
    def __init__(self, discount_rate=1.0): 
        self.discount_rate = discount_rate
    _transition_rewards = {
        's1': {
            "a1": {
                's1': (1, 3)
            },
            "a2": {
                's2': (1, 1)
            }
        },
        's2': {
            "a1": {
                "s2": (1, 0)
            },
            "a2": {
                "s3": (1, 1)
            }
        },
        "s3": {
            "a1": {
                "s3": (1, 2)
            }
        }
    }
    def initial_state_dist(self):
        return DictDistribution({'s1': 1})
    def is_absorbing(self, s):
        return False
    def optimal_policy(self):
        return {
            "s1": DictDistribution({'a1': 1}),
            "s2": DictDistribution({'a2': 1}),
            "s3": DictDistribution({'a1': 1})
        }
    def optimal_state_value(self):
        if self.discount_rate < 1.0:
            return {
                "s1": self.reward("s1", "a1", "s1")*(1/(1 - self.discount_rate)),
                "s2": self.reward("s2", "a2", "s3") + \
                    self.discount_rate*self.reward("s3", "a1", "s3")*(1/(1 - self.discount_rate)),
                "s3": self.reward("s3", "a1", "s3")*(1/(1 - self.discount_rate))
            }
        # technically this should be up to an additive constant
        return {
            "s1": 0,
            "s2": -1,
            "s3": 0
        }
    def optimal_state_gain(self):
        if self.discount_rate < 1.0:
            return {
                "s1": 0,
                "s2": 0,
                "s3": 0
            }
        return {
            "s1": 3,
            "s2": 2,
            "s3": 2,
        }

class AbsorbingStateTester(TabularMarkovDecisionProcess):
    def __init__(
        self,
        discount_rate=1.0,
        last_reward=0,
        last_actions=(0, ),
        explicit_absorbing_flag=False
    ): 
        self.discount_rate = discount_rate
        self.last_reward = last_reward
        self.last_actions = last_actions
        self.explicit_absorbing_flag = explicit_absorbing_flag
    def initial_state_dist(self):
        return DictDistribution({0: 1})
    def actions(self, s):
        if s == 2:
            return self.last_actions
        return (0, 1)
    def next_state_dist(self, s, a):
        assert s + a <= 2
        return DictDistribution({s + a: 1})
    def reward(self, s, a, ns):
        if ns == 2:
            return self.last_reward
        return 0
    def is_absorbing(self, s) -> bool:
        if self.explicit_absorbing_flag and s == 2:
            return True
        return False 

class TiedPaths(TransitionRewardDictMDP, TestDomain):
    """
    Acyclic MDP with 3 paths that are tied in
    value but in different ways:
    - -2 reward in a single transition (path A)
    - -2 reward in 2 transitions assuming no discounting (path B)
    - -2 reward in expectation (paths C)
    """
    _transition_rewards = {
        'start': {
            'a': {
                'a': (1, -2)
            },
            'b': {
                'b1': (1, -1)
            },
            'c': {
                'c1': (1/3, -4),
                'c2': (2/3, -1),
            }
        },
        'b1': {
            'b': {
                'b2': (1, -1)
            }
        },
        'a': {'end': {'end': (1, 0)}},
        'b2': {'end': {'end': (1, 0)}},
        'c1': {'end': {'end': (1, 0)}},
        'c2': {'end': {'end': (1, 0)}},
        'end': {'end': {'end': (1, 0)}},
    }
    def __init__(self, discount_rate=.95):
        self.discount_rate = discount_rate
    def initial_state_dist(self):
        return DictDistribution({'start': 1})
    def is_absorbing(self, s):
        return s == 'end'
    def optimal_policy(self):
        basic_policy = {
            'a': DictDistribution({'end': 1}),
            'b2': DictDistribution({'end': 1}),
            'b1': DictDistribution({'b': 1}),
            'c1': DictDistribution({'end': 1}),
            'c2': DictDistribution({'end': 1}),
            'end': DictDistribution({'end': 1}),
        }
        if self.discount_rate == 1.0:
            return {
                **basic_policy,
                'start': DictDistribution({
                    'a': 1/3,
                    'b': 1/3,
                    'c': 1/3,
                })
            }
        elif 0 <= self.discount_rate < 1.0:
            return {
                **basic_policy,
                'start': DictDistribution({'b': 1})
            }
        else:
            raise ValueError("Invalid discount rate")

    def optimal_state_value(self):
        return {
            'start': -(1 + self.discount_rate),
            'a': 0,
            'b1': -1,
            'b2': 0,
            'c1': 0,
            'c2': 0,
            'end': 0
        }

class RussellNorvigGrid(GridWorld, TabularMarkovDecisionProcess):
    # g is goal, x is lava, # is the wall, and s in the agent's starting location
    grid_string = '''
        ...g
        .#.x
        s...
    '''
    grid = [list(r.strip()) for r in grid_string.split('\n') if len(r.strip()) > 0]

    # states are xy tuples indexed at 0
    loc_to_feature = {}
    for y, row in enumerate(grid[::-1]):
        for x, c in enumerate(row):
            loc_to_feature[(x, y)] = c
    _locFeatures = loc_to_feature
    _walls = [s for s, f in loc_to_feature.items() if f == '#']
    _initStates = [s for s, f in loc_to_feature.items() if f == 's']
    _absorbingStates = [s for s, f in loc_to_feature.items() if f in 'gx']
            
    def __init__(
        self,
        discount_rate=1.0,
        slip_prob=0.2,
        step_reward=-0.04
    ):
        self.discount_rate = discount_rate
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self._states = tuple(sorted(self.loc_to_feature.keys()))
        self._width = max([x for x, y in self._states]) + 1
        self._height = max([y for x, y in self._states]) + 1
    def is_absorbing(self, s):
        return self.loc_to_feature[s] in 'gx'
    def actions(self, s):
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    def initial_state_dist(self):
        s0 = [s for s, f in self.loc_to_feature.items() if f == 's']
        return DictDistribution.uniform(s0)
    def reward(self, s, a, ns):
        # standard reward function assigns +1 for goal, -1 for lava, -0.04 for all other states
        r = self.step_reward
        if self.loc_to_feature.get(ns, '') == 'g':
            r += 1.0
        elif self.loc_to_feature.get(ns, '') == 'x':
            r += -1.0
        return r
        # raise Exception("Invalid state for reward", ns)
    def is_valid_loc(self, s):
        return (s in self.loc_to_feature) and (self.loc_to_feature[s] != '#')
    def move(self, s, a):
        ns = (s[0] + a[0], s[1] + a[1])
        # 'bouncing' if the next state is a wall or off the grid
        if not self.is_valid_loc(ns):
            return s
        return ns
    def next_state_dist(self, s, a):
        is_x_move = a[0] != 0
        slip_op1 = (0, -1) if is_x_move else (-1, 0)
        slip_op2 = (0, 1) if is_x_move else (1, 0)
        
        ns_dist = defaultdict(float)
        intended_ns = self.move(s, a)
        ns_dist[intended_ns] += 1 - self.slip_prob

        # adding the slips to the next state distribution
        slip_ns1 = self.move(s, slip_op1)
        ns_dist[slip_ns1] += self.slip_prob / 2
        slip_ns2 = self.move(s, slip_op2)
        ns_dist[slip_ns2] += self.slip_prob / 2
        return DictDistribution(ns_dist)
        
class RussellNorvigGrid_Fig17_3(RussellNorvigGrid, TestDomain):
    """
    Slippery gridworld with parameters from AIMA (3rd Ed)
    Figure 17.2 and 17.3.
    """
    def __init__(self, discount_rate=1.0) -> None:
        super().__init__(
            discount_rate=discount_rate,
            slip_prob=0.2,
            step_reward=-0.04
        )
    def optimal_policy(self):
        assert self.discount_rate == 1.0
        return {
            (0, 0): DictDistribution({(0, 1): 1}),
            (0, 1): DictDistribution({(0, 1): 1}),
            (0, 2): DictDistribution({(1, 0): 1}),
            (1, 0): DictDistribution({(-1, 0): 1}),
            (1, 1): DictDistribution({(0, 1): 1}),
            (1, 2): DictDistribution({(1, 0): 1}),
            (2, 0): DictDistribution({(-1, 0): 1}),
            (2, 1): DictDistribution({(0, 1): 1}),
            (2, 2): DictDistribution({(1, 0): 1}),
            (3, 0): DictDistribution({(-1, 0): 1}),
            (3, 1): DictDistribution({a: 1/len(self.actions((3, 1))) for a in self.actions((3, 1))}),
            (3, 2): DictDistribution({a: 1/len(self.actions((3, 2))) for a in self.actions((3, 2))}),
        } 
    def optimal_state_value(self):
        assert self.discount_rate == 1.0
        return {
            (0, 0): .705,
            (0, 1): .762,
            (0, 2): .812,
            (1, 0): .655,
            (1, 1): .868*.8 + .762*.1 + .660*.1 - 0.04,
            (1, 2): .868,
            (2, 0): .611,
            (2, 1): .660,
            (2, 2): .918,
            (3, 0): .388,
            (3, 1): 0.,
            (3, 2): 0.,
        }
