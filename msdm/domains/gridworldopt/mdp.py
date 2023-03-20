import pyximport
pyximport.install(
    language_level=3,
)
import numpy as np
from msdm.domains.gridworldopt.dynamics import transition_reward_matrices
from msdm.domains.gridmdp import GridMDP, Location, GridAction
from msdm.core.mdp.tabularmdp import FromVectorizedMixIn
from msdm.core.table import domaintuple
from msdm.core.utils.funcutils import cached_property, method_cache
import dataclasses
from itertools import product

@dataclasses.dataclass
class GridWorld(GridMDP,FromVectorizedMixIn):
    feature_map : list
    feature_rewards : dict = None
    absorbing_features : list = ("g",)
    wall_features : list = ("#",)
    initial_features : list = ("s",)
    step_cost : float = -1
    wall_bump_cost : float = -10
    stay_prob : float = 0.0
    left_slip_prob : float = 0.0
    right_slip_prob : float = 0.0
    back_slip_prob : float = 0.0
    wait_action : bool = True
    discount_rate : float = 1.0

    @cached_property
    def absorbing_state_vec(self):
        vec = map_to_xy_array(
            np.isin(self.feature_map_array, self.absorbing_features)
        ).flatten()
        return make_immutable(vec)
    
    @cached_property
    def initial_state_vec(self):
        vec = map_to_xy_array(
            np.isin(self.feature_map_array, self.initial_features)
        ).flatten()
        return make_immutable(vec)

    @cached_property
    def action_matrix(self) -> np.ndarray:
        action_matrix = np.ones((len(self.state_list), len(self.action_list)))
        return make_immutable(action_matrix)

    @cached_property
    def transition_matrix(self) -> np.ndarray:
        transition_matrix, _ = self.transition_reward_matrices()
        return transition_matrix
    @cached_property
    def state_action_reward_matrix(self) -> np.ndarray:
        _, state_action_reward_matrix = self.transition_reward_matrices()
        return state_action_reward_matrix
    @cached_property
    def reward_matrix(self) -> np.ndarray:
        reward_matrix = np.einsum(
            "ijk,ij->ijk",
            self.transition_matrix > 0,
            self.state_action_reward_matrix
        )
        return make_immutable(reward_matrix)

    @method_cache
    def transition_reward_matrices(self):
        xy_walls = map_to_xy_array(self.wall_map)
        xy_rewards = map_to_xy_array(self.reward_map)
        state_list = np.array(self.state_list)
        action_list = np.array(self.action_list)
        n_actions = len(action_list)
        transition_matrix = np.zeros(
            (self.width, self.height , n_actions, self.width, self.height)
        ).astype(np.double)
        reward_matrix = np.zeros(
            (self.width, self.height, n_actions)
        ).astype(np.double)

        transition_reward_matrices(
            step_cost=-1,  
            wall_bump_cost=-10,
            stay_prob=0.01,
            left_slip_prob=0.1,
            right_slip_prob=0.1,
            back_slip_prob=0.04,
            state_list=state_list.astype(np.intc), # x, y
            action_list=action_list.astype(np.intc), # dx, dy
            xy_walls=xy_walls.astype(np.intc), # x, y
            xy_rewards=xy_rewards.astype(np.double), # x, y
            transition_matrix=transition_matrix, # x, y, ai, nx, ny
            reward_matrix=reward_matrix, # x, y, ai
        )
        transition_matrix = transition_matrix.reshape(
            len(state_list), len(action_list), len(state_list)
        )
        reward_matrix = reward_matrix.reshape(
            len(state_list), len(action_list)
        )
        transition_matrix = make_immutable(transition_matrix)
        reward_matrix = make_immutable(reward_matrix)
        return transition_matrix, reward_matrix

    def plot(
            self,
            ax=None,
            feature_colors=None,
            feature_markers=None,
        ):
        if feature_colors is None:
            feature_colors = {
                '.': 'white',
                '#': (.2, .2, .2),
                '$': 'yellow'
            }
            abs_max_reward = max([abs(v) for v in self.feature_rewards.values()])
            for f, r in self.feature_rewards.items():
                if r < 0:
                    feature_colors[f] = (1, 0, 0, abs(r)/abs_max_reward)
                elif r > 0:
                    feature_colors[f] = (0, .6, 0, abs(r)/abs_max_reward)
        plotter = super().plot(
            ax=ax,
            feature_colors=feature_colors,
            feature_markers={}
        )
        if feature_markers is None:
            feature_markers = {
                '@': 'o',
                '$': '*'
            }
        plotter.mark_feature(
            '@', marker='o',
            plot_kwargs=dict(
                markeredgecolor='k',
                markerfacecolor='blue',
                fillstyle='full',
            )
        )
        plotter.mark_feature(
            '$', marker='*',
            plot_kwargs=dict(
                markeredgecolor='k',
                markerfacecolor='yellow',
                fillstyle='full',
                markersize=20
            )
        )
        return plotter

    @cached_property
    def _grid_string(self):
        return "\n".join("".join(row) for row in self.feature_map)

    @cached_property
    def feature_map_array(self):
        return make_immutable(
            np.array([list(r) for r in self.feature_map])
        )
    @cached_property
    def wall_map(self):
        return make_immutable(
            np.isin(self.feature_map_array, self.wall_features)
        )
    @cached_property
    def reward_map(self):
        reward_map = np.zeros_like(self.feature_map_array, dtype=float)
        for feature, reward in self.feature_rewards.items():
            reward_map[self.feature_map_array == feature] = reward
        return make_immutable(reward_map)
    @cached_property
    def height(self):
        return len(self.feature_map)
    @cached_property
    def width(self):
        return len(self.feature_map[0])
    @cached_property
    def state_list(self):
        return domaintuple([
            Location(x, y) for x, y in product(range(self.width), range(self.height))
        ])
    @cached_property
    def action_list(self):
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.wait_action:
            actions.append((0, 0))
        return domaintuple([
            GridAction(dx, dy) for dx, dy in actions
        ])


def make_immutable(arr: np.ndarray) -> np.ndarray: 
    arr.setflags(write=False)
    return arr

def map_to_xy_array(arr: np.ndarray) -> np.ndarray:
    return arr[::-1].T

def xy_array_to_map(xy_arr: np.ndarray) -> np.ndarray:
    return xy_arr.T[::-1]