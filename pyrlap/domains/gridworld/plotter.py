import copy

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

from pyrlap.domains.gridworld import GridWorld
from pyrlap.domains.gridworld.gridworldvis import visualize_states, \
    visualize_action_values, plot_agent_location, plot_text, \
    visualize_walls, visualize_trajectory
from pyrlap.core.agent import Agent, ValueFunction
from pyrlap.algorithms.valueiteration import ValueIteration

DISTINCT_COLORS = [
    '#A9A9A9', '#e6194b', '#3cb44b',
    '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080',
    '#e6beff', '#9a6324', '#fffac8',
    '#800000', '#aaffc3', '#808000',
    '#ffd8b1', '#000075', '#808080',
    '#ffffff', '#000000'
]

def get_contrast_color(color):
    r, g, b = colors.to_rgb(color)
    luminance = (0.299*r**2 + 0.587*g**2 + 0.114*b**2)**.5
    if luminance < .7:
        return "white"
    return 'grey'

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
            'n': 'white',
            '#': 'black',
            'j': 'lightgreen'
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

        states_to_plot = []
        for s in gw.states:
            if gw.is_any_terminal(s):
                continue
            states_to_plot.append(s)
            if s in tile_colors:
                continue
            if (s, None) in gw.walls:
                continue
            f = gw.state_features.get(s, '.')
            tile_colors[s] = feature_colors.get(f, 'grey')


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

        self.states_to_plot = states_to_plot
        self.annotations = {}
        self.trajectories = {}

    def plot(self):
        visualize_states(ax=self.ax, states=self.states_to_plot,
                         tile_color=self.tile_colors)
        visualize_walls(ax=self.ax, walls=self.gw.walls)
        return self

    def plot_policy(self, agent: Agent = None, policy_dict : dict = None):
        if policy_dict is None:
            p = agent.to_dict()
        else:
            p = policy_dict
        p = {s: adist for s, adist in p.items() if not self.gw.is_wall(s)}
        visualize_action_values(ax=self.ax, state_action_values=p,
                                global_maxval=1.0)
        return self

    def plot_action_values(self, action_values, **kwargs):
        av = {s: vs for s, vs in action_values.items() if not self.gw.is_wall(s)}
        visualize_action_values(ax=self.ax,
                                state_action_values=av,
                                **kwargs)
        return self

    def plot_value(self,
                   vi : ValueIteration = None,
                   vf : ValueFunction = None,
                   show_value_numbers : bool = True,
                   fontsize=10,
                   cmap : "matplotlib color map" = "bwr_r",
                   value_function_range = None
                   ):
        if vi is not None:
            visualize_action_values(ax=self.ax,
                                    state_action_values=vi.action_value_function,
                                    color_valence=True
                                    )
        elif vf is not None:
            vmax_abs = max(abs(v) for v in vf.values())

            if value_function_range is None:
                value_function_range = [-vmax_abs, vmax_abs]
            vmin, vmax = value_function_range

            colorrange = plt.get_cmap(cmap)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            value_map = cmx.ScalarMappable(norm=color_norm, cmap=colorrange)
            tile_colors = {s: value_map.to_rgba(v) for s, v in vf.items()}
            states_to_plot = \
                [s for s in self.states_to_plot if not self.gw.is_wall(s)]
            visualize_states(ax=self.ax, states=states_to_plot,
                             tile_color=tile_colors)
            if show_value_numbers:
                for s in self.gw.get_states():
                    if self.gw.is_any_terminal(s):
                        continue
                    if self.gw.is_wall(s):
                        continue
                    if s not in vf:
                        continue
                    self.annotate(x=s[0],
                                  y=s[1],
                                  text="%.2f" % vf[s],
                                  color=get_contrast_color(tile_colors[s]),
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  fontsize=fontsize
                                  )
        return self

    def plot_trajectory(self, traj, name=None, **kwargs):
        traj_patches = visualize_trajectory(axis=self.ax, traj=traj, **kwargs)
        if name is None:
            name = "trajectory-"+str(len(self.trajectories))
        self.trajectories[name] = traj_patches
        return self

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
        return self

    def plot_agent(self, s=None):
        if s is None:
            s = self.gw.get_init_state()
        self.agent = plot_agent_location(s, ax=self.ax)
        return self

    def title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)
        return self