import copy

from pyrlap.domains.gridworld import GridWorld
from pyrlap.domains.gridworld.gridworldvis import visualize_states, \
    visualize_action_values, plot_agent_location, plot_text, \
    visualize_walls, visualize_trajectory

import matplotlib.pyplot as plt

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

        plot_states = []
        for s in gw.states:
            if gw.is_any_terminal(s):
                continue
            if s in tile_colors:
                continue
            f = gw.state_features.get(s, '.')
            tile_colors[s] = feature_colors[f]
            plot_states.append(s)

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