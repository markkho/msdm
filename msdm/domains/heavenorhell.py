import copy
from collections import namedtuple, defaultdict
from msdm.core.pomdp import TabularPOMDP
from msdm.core.distributions import DictDistribution

State = namedtuple("State", "x y heaven hell")
Action = namedtuple("Action", "dx dy read")
Observation = namedtuple("Observation", "x y heaven")

class HeavenOrHell(TabularPOMDP):
    def __init__(
        self,
        coherence=.95,
        discount_rate=.95,
        step_cost=-1,
        heaven_reward=50,
        hell_reward=-50,
        grid=None
    ):
        """
        Heaven or Hell (a.k.a. information gathering) as first described by
        [Bonet and Geffner (1998)](https://bonetblai.github.io/reports/fall98-pomdp.pdf).

        A simple POMDP where the agent must gather information to figure out
        which goal is gives a reward or punishment.

        Parameters
        ---------
        :coherence:       The strength of the signal about which side is heaven/hell
        :discount_rate:
        :step_cost:       Step cost when not reading
        :heaven_reward:
        :hell_reward:
        :grid:            A multiline string representing a heaven/hell configuration.
                          `s` is the initial state,
                          `#` are walls,
                          `h` and `g` are potential heaven/hell locations, and
                          `c` is where you go to learn about how to get to heaven/hell.
        """
        if grid is None:
            grid = \
            """
            h.g
            #.#
            #sc
            """
        grid = [list(r.strip()) for r in grid.split('\n') if len(r.strip()) > 0]
        self.grid = grid
        self.loc_features = {}
        self.features_loc = defaultdict(list)
        for y, row in enumerate(grid):
            for x, f in enumerate(row):
                self.loc_features[(x, y)] = f
                self.features_loc[f].append((x, y))
        self.coherence = coherence
        self.discount_rate = discount_rate
        self.step_cost = step_cost
        self.heaven_reward = heaven_reward
        self.hell_reward = hell_reward

    def initial_state_dist(self):
        x, y = self.features_loc['s'][0]
        return DictDistribution({
            State(x=x, y=y, heaven='g', hell='h'): .5,
            State(x=x, y=y, heaven='h', hell='g'): .5,
        })

    def actions(self, s):
        return (
            Action(0, -1, False),
            Action(0, 1, False),
            Action(-1, 0, False),
            Action(1, 0, False),
            Action(0, 0, True),
        )

    def is_absorbing(self, s):
        loc = (s.x, s.y)
        return self.loc_features[loc] in (s.heaven, s.hell)

    def next_state_dist(self, s, a):
        x, y = s.x, s.y
        nx, ny = (s.x + a.dx, s.y + a.dy)
        if self.loc_features.get((nx, ny), '#') == '#':
            nx, ny = (s.x, s.y)
        return DictDistribution({
            State(x=nx, y=ny, heaven=s.heaven, hell=s.hell): 1
        })

    def reward(self, s, a, ns):
        r = 0
        r += self.step_cost
        if self.loc_features[(ns.x, ns.y)] == ns.heaven:
            r += self.heaven_reward
        elif self.loc_features[(ns.x, ns.y)] == ns.hell:
            r += self.hell_reward
        return r

    def observation_dist(self, a, ns):
        nloc = ns.x, ns.y
        if a.read and (self.loc_features[nloc] == 'c'): #go to church
            return DictDistribution({
                Observation(x=ns.x, y=ns.y, heaven=ns.heaven): self.coherence,
                Observation(x=ns.x, y=ns.y, heaven=ns.hell): 1 - self.coherence
            })
        return DictDistribution({
                Observation(x=ns.x, y=ns.y, heaven=" "): 1.,
        })

    def state_string(self, s):
        grid = copy.deepcopy(self.grid)
        for y, row in enumerate(grid):
            for x, f in enumerate(row):
                if (x, y) == (s.x, s.y):
                    grid[y][x] = '@'
        return '\n'.join([''.join(r) for r in grid])
