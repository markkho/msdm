import numpy as np
from collections import defaultdict
from msdm.domains.gridmdp import GridMDP
from msdm.core.distributions import DictDistribution
from msdm.core.utils.funcutils import cached_property

def rotation_matrix(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ]).round(10)

class SlipperyMaze(GridMDP):
    wall_features = "#"
    goal_features = "$"
    lava_features = "x"
    lava_cost = -100
    initial_features = "@"
    step_cost = -1
    rotation_matrices = dict(
        left=rotation_matrix(np.pi/2).astype(int),
        back=rotation_matrix(np.pi).astype(int),
        right=rotation_matrix(3*np.pi/2).astype(int),
    )
    
    def __init__(
        self,
        tile_array,
        left_slip_prob=.05,
        right_slip_prob=.05,
        back_slip_prob=.05,
        stay_prob=.05,
        discount_rate=1.0
    ):
        super().__init__('\n'.join(tile_array))
        self.tile_array = np.array([list(row) for row in tile_array])
        self.move_probs = dict(
            left=left_slip_prob,
            right=right_slip_prob,
            back=back_slip_prob,
            stay=stay_prob,
        )
        self.move_probs['forward'] = 1 - sum(self.move_probs.values())
        self.discount_rate = discount_rate
    def initial_state_dist(self):
        return DictDistribution.uniform(self.initial_locations)
    def actions(self, s):
        return tuple([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
    def in_grid(self, s):
        return 0 <= s[0] < self.width and 0 <= s[1] < self.height
    def next_state_dist(self, s, a):
        ns_dist = defaultdict(float)
        moves = dict(
            forward = a,
            left = self.rotation_matrices['left']@a,
            right = self.rotation_matrices['right']@a,
            back = self.rotation_matrices['back']@a,
            stay = (0, 0),
        )
        for move_name, move in moves.items():
            ns = (s[0]+move[0], s[1]+move[1])
            if ns in self.wall_locations:
                ns = s
            elif not self.in_grid(ns):
                ns = s
            ns_dist[ns] += self.move_probs[move_name]
        return DictDistribution(ns_dist)
    @cached_property
    def state_list(self):
        return self.location_list
    def reward(self, s, a, ns):
        if ns in self.lava_locations:
            return self.lava_cost
        return self.step_cost
    def is_absorbing(self, s):
        return s in self.absorbing_locations
    def locations_with(self, features):
        return tuple(sorted(zip(*np.where(np.isin(self.tile_array[::-1,:].T, tuple(features))))))
    @cached_property
    def wall_locations(self):
        return self.locations_with(self.wall_features)
    @cached_property
    def initial_locations(self):
        return self.locations_with(self.initial_features)
    @cached_property
    def goal_locations(self):
        return self.locations_with(self.goal_features)
    @cached_property
    def lava_locations(self):
        return self.locations_with(self.lava_features)
    @cached_property
    def absorbing_locations(self):
        return self.lava_locations + self.goal_locations
    def plot(self):
        return super().plot(
            feature_colors={
                'x': 'red',
                '@': 'w',
                '$': 'green',
                '#': 'k',
                '.': 'w',
            },
            feature_markers={
                '@': 'o',
                '$': 'x',
                'x': 'x'
            }
        )