from collections import namedtuple
from msdm.core.pomdp import TabularPOMDP
from msdm.core.distributions import DictDistribution

State = namedtuple("State", "location is_loaded")
Action = namedtuple("Action", "dlocation")

class LoadUnload(TabularPOMDP):
    '''
    The load/unload task, following specification in [1]. This task is an infinite-horizon 1-dimensional grid-world
    where the agent starts at the location marked U below. The agent receives a reward every time it visits
    the location U after having visited location L.

    U......L

    [1] Meuleau et al. (1999). Solving POMDPs by Searching the Space of Finite Policies. https://arxiv.org/abs/1301.6720
    '''

    action_list = [Action(-1), Action(+1)]
    observation_list = ['load', 'unload', 'other']

    def __init__(self, *, nstates=8, discount_rate=0.99):
        self.nstates = nstates
        self.discount_rate = discount_rate

    def initial_state_dist(self):
        return DictDistribution.deterministic(State(0, False))

    def is_absorbing(self, s):
        return False

    def actions(self, s):
        return self.action_list

    def next_state_dist(self, s, a):
        location, is_loaded = s
        location = min(max(location + a.dlocation, 0), self.nstates-1)

        if location == 0:
            is_loaded = False
        if location == self.nstates - 1:
            is_loaded = True

        return DictDistribution.deterministic(State(location, is_loaded))

    def reward(self, s, a, ns):
        # Rewarded when a load is dropped off!
        if s.is_loaded and not ns.is_loaded:
            return +1
        return 0

    def observation_dist(self, a, ns):
        if ns.location == 0:
            o = 'unload'
        elif ns.location == self.nstates - 1:
            o = 'load'
        else:
            o = 'other'
        return DictDistribution.deterministic(o)
