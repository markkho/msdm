import copy

from pyrlap.domains.gridworld import GridWorld
from pyrlap.domains.gridworld.gridworldvis import visualize_states, \
    visualize_action_values, plot_agent_location, plot_text, \
    visualize_walls, visualize_trajectory
from pyrlap.core.agent import Agent
from pyrlap.algorithms.valueiteration import ValueIteration

import matplotlib.pyplot as plt

DISTINCT_COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                   '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                   '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                   '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                   '#ffffff', '#000000']


class GridWorldPlotter(object):
    def __init__(self,
                 gw : GridWorld,
                 tile_colors : dict = None,
                 feature_colors : dict = None,
                 ax : plt.Axes = None,
                 figsize: tuple = None,
                 title: str = None
                 ):
        default_feature_colors = {
            'a': 'orange',
            'b': 'purple',
            'c': 'cyan',
            'x': 'red',
            'p': 'pink',
            '.': 'white',
            'y': 'yellow',
            'g': 'yellow',
            'n': 'white'
        }
        if feature_colors is None:
            feature_colors = default_feature_colors
        else:
            temp_fcolors = copy.deepcopy(default_feature_colors)
            temp_fcolors.update(feature_colors)
            feature_colors = temp_fcolors

        if tile_colors is None:
            tile_colors = {}
        else:
            tile_colors = copy.copy(tile_colors)

        plot_states = []
        for s in gw.states:
            if gw.is_any_terminal(s):
                continue
            plot_states.append(s)
            if s in tile_colors:
                continue
            if (s, None) in gw.walls:
                continue
            f = gw.state_features.get(s, '.')
            tile_colors[s] = feature_colors[f]


        if figsize is None:
            figsize = (5, 5)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if title is not None:
            ax.set_title(title)

        self.gw = gw
        self.feature_colors = feature_colors
        self.tile_colors = tile_colors
        self.ax = ax

        self.plot_states = plot_states
        self.annotations = {}
        self.trajectories = {}

    def plot(self):
        visualize_states(ax=self.ax, states=self.plot_states,
                         tile_color=self.tile_colors)
        visualize_walls(ax=self.ax, walls=self.gw.walls)

    def plot_policy(self, agent: Agent):
        p = agent.to_dict()
        visualize_action_values(ax=self.ax, state_action_values=p)

    def plot_value(self, vi: ValueIteration):
        visualize_action_values(ax=self.ax,
                                state_action_values=vi.action_value_function,
                                color_valence=True
                                )

    def plot_trajectory(self, traj, name=None, **kwargs):
        traj_patches = visualize_trajectory(axis=self.ax, traj=traj, **kwargs)
        if name is None:
            name = "trajectory-"+str(len(self.trajectories))
        self.trajectories[name] = traj_patches

    def annotate(self, x, y, text,
                 outline=False,
                 outline_linewidth=1,
                 outline_color='black',
                 name=None,
                 **kwargs):
        txt = plot_text(axis=self.ax,
                        state=(x, y),
                        text=text,
                        outline=outline,
                        outline_linewidth=outline_linewidth,
                        outline_color=outline_color,
                        **kwargs)
        if name is None:
            name = "annotation-"+str(len(self.annotations))
        self.annotations[name] = txt

    def plot_agent(self, s=None):
        if s is None:
            s = self.gw.get_init_state()
        self.agent = plot_agent_location(s, ax=self.ax)

    def title(self, title):
        self.ax.set_title(title)