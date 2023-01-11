from multiprocessing.sharedctypes import Value
import numpy as np
from frozendict import frozendict
from typing import Mapping, Union, Callable, Hashable
from numbers import Number

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patheffects as path_effects

from msdm.core.mdp.tabularpolicy import TabularPolicy
from msdm.domains.gridworld.mdp import GridWorld


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
    luminance = (0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2) ** .5
    if luminance < .7:
        return "white"
    return 'grey'


class GridWorldPlotter:
    def __init__(self, gw: GridWorld, ax: plt.Axes):
        self.gw = gw
        self.ax = ax
        self.ax.axis('off')
        self.ax.set_xlim(-0.1, self.gw.width + .1)
        self.ax.set_ylim(-0.1, self.gw.height + .1)
        # self.ax.axis('equal')
        ax.set_aspect('equal')
    
    def _get_state_xy(self, s):
        if isinstance(s, (dict, frozendict)):
            xy = (s['x'], s['y'])
        elif isinstance(s, (tuple, list)):
            xy = s
        else:
            raise ValueError("Unrecognized grid state representation")
        return xy

    def plot_features(self, featurecolors, edgecolor='darkgrey') -> "GridWorldPlotter":
        """Plot gridworld features"""
        ss = self.gw.state_list
        for s in ss:
            x, y = self._get_state_xy(s)
            if x < 0 or y < 0:
                continue
            f = self.gw._locFeatures.get(s, '.')[0]
            color = featurecolors.get(f, 'w')
            square = Rectangle((x, y), 1, 1,
                               facecolor=color,
                               edgecolor=edgecolor,
                               linewidth=2)
            self.ax.add_patch(square)
        return self

    def plot_outer_box(self):
        outerbox = Rectangle((0, 0), self.gw.width, self.gw.height,
                             fill=False, edgecolor='black',
                             linewidth=2)
        self.ax.add_patch(outerbox)
        return self

    def plot_walls(self, facecolor='k', edgecolor='darkgrey'):
        for ws in self.gw.walls:
            xy = self._get_state_xy(ws)
            square = Rectangle(xy, 1, 1,
                               facecolor=facecolor,
                               edgecolor=edgecolor,
                               linewidth=2)
            self.ax.add_patch(square)
        return self

    def plot_initial_states(self, markersize=15):
        for s in self.gw.initial_states:
            xy = self._get_state_xy(s)
            x, y = xy 
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor='cornflowerblue',
                         marker='o',
                         markersize=markersize,
                         markeredgewidth=2,
                         fillstyle='none')
        return self

    def plot_absorbing_states(self, markersize=15):
        for s in self.gw.absorbing_states:
            xy = self._get_state_xy(s)
            x, y = xy 
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor='cornflowerblue',
                         marker='x',
                         markersize=markersize,
                         markeredgewidth=2)

    def plot_trajectory(self,
                        state_traj,
                        action_traj=None,  # not implemented yet
                        color='k',
                        outline=False,
                        outlinecolor='w',
                        jitter_mean=0,
                        jitter_var=.1,
                        end_jitter=False,
                        linewidth=1,
                        **kwargs) -> "GridWorldPlotter":
        if action_traj is not None:
            assert len(state_traj) == len(action_traj)

        xys = []
        for s in state_traj:
            if isinstance(s, (tuple, list)):
                if s[0] < 0 or s[1] < 0:
                    continue
                xys.append(s)
            elif isinstance(s, (dict, frozendict)):
                if s['x'] < 0 or s['y'] < 0:
                    continue
                xys.append((s['x'], s['y']))

        if len(xys) == 2:
            p0 = tuple(np.array(xys[0]) + .5)
            p2 = tuple(np.array(xys[1]) + .5)
            p1 = np.array([(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]) \
                 + np.random.normal(0, jitter_var, 2)
            if end_jitter:
                p0 = tuple(
                    np.array(p0) + np.random.normal(jitter_mean, jitter_var, 2))
                p1 = tuple(
                    np.array(p1) + np.random.normal(jitter_mean, jitter_var, 2))
            segments = [[p0, p1, p2], ]
        elif (len(xys) == 3) and (xys[0] == xys[2]):
            p0 = tuple(np.array(xys[0]) + .5)
            p2 = tuple(np.array(xys[1]) + .5)
            if abs(p0[0] - p2[0]) > 0:  # horizontal
                jitter = np.array(
                    [0, np.random.normal(jitter_mean, jitter_var * 2)])
                p2 = p2 - np.array([.25, 0])
            else:  # vertical
                jitter = np.array(
                    [np.random.normal(jitter_mean, jitter_var * 2), 0])
                p2 = p2 - np.array([0, .25])
            p1 = p2 + jitter
            p3 = p2 - jitter
            segments = [[p0, p1, p2], [p2, p3, p0]]
        else:
            state_coords = []
            for xy in xys:
                jitter = np.random.normal(jitter_mean, jitter_var, 2)
                coord = np.array(xy) + .5 + jitter
                state_coords.append(tuple(coord))
            if not end_jitter:
                state_coords[0] = tuple(np.array(xys[0]) + .5)
                state_coords[-1] = tuple(np.array(xys[-1]) + .5)
            join_point = state_coords[0]
            segments = []
            for i, xy in enumerate(state_coords[:-1]):
                nxy = state_coords[i + 1]

                segment = []
                segment.append(join_point)
                segment.append(xy)
                if i < len(xys) - 2:
                    join_point = tuple(np.mean([xy, nxy], axis=0))
                    segment.append(join_point)
                else:
                    segment.append(nxy)
                segments.append(segment)

        outline_patches = []
        if outline:
            for segment, step in zip(segments, xys[:-1]):
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(segment, codes)
                outline_patch = patches.PathPatch(path, facecolor='none',
                                                  capstyle='butt',
                                                  edgecolor=outlinecolor,
                                                  linewidth=linewidth * 2)
                self.ax.add_patch(outline_patch)
                outline_patches.append(outline_patch)

        xys_patches = []
        for segment, step in zip(segments, xys[:-1]):
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(segment, codes)

            patch = patches.PathPatch(path, facecolor='none', capstyle='butt',
                                      edgecolor=color, linewidth=linewidth,
                                      **kwargs)
            xys_patches.append(patch)
            self.ax.add_patch(patch)
        return self

    def plot_state_map(self,
                       state_map: Mapping,
                       plot_over_walls=False,
                       fontsize=10,
                       show_numbers=True,
                       value_range=None,
                       show_colors=True,
                       is_categorical=False,
                       color_value_func="bwr_r") -> "GridWorldPlotter":
        if len(state_map) == 0:
            return self
        # state map - colors / numbers
        vmax_abs = max(abs(v) for k, v in state_map.items())
        if value_range is None:
            value_range = [-vmax_abs, vmax_abs]
        vmin, vmax = value_range
        if is_categorical:
            color_value_func = lambda i: DISTINCT_COLORS[
                int(i) % len(DISTINCT_COLORS)]
        elif isinstance(color_value_func, str):
            colorrange = plt.get_cmap(color_value_func)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            color_value_map = cmx.ScalarMappable(norm=color_norm,
                                                cmap=colorrange)
            color_value_func = lambda v: color_value_map.to_rgba(v)
        for s, v in state_map.items():
            if self.gw.is_absorbing(s):
                continue
            if (not plot_over_walls) and (s in self.gw.walls):
                continue
            if isinstance(s, (dict, frozendict)):
                xy = s['x'], s['y']
            elif isinstance(s, tuple) or isinstance(s, list):
                xy = s
            else:
                raise Exception("unknown state representation")

            color = 'w'
            if show_colors:
                color = color_value_func(v)
                square = Rectangle(xy, 1, 1,
                                   color=color,
                                   ec='k', lw=2)
                self.ax.add_patch(square)
            if show_numbers:
                self.ax.text(xy[0] + .5, xy[1] + .5,
                             f"{v : .2f}",
                             fontsize=fontsize,
                             color=get_contrast_color(color),
                             horizontalalignment='center',
                             verticalalignment='center')
        return self

    def plot_state_action_map(self,
                              state_action_map: Mapping,
                              plot_over_walls=False,
                              value_range=None,
                              color_value_func: Union[Callable, str] = "bwr_r",
                              arrow_width=.1,
                              show_numbers=False,
                              numbers_kw=None,
                              visualization_type="arrow"
                              ) -> "GridWorldPlotter":
        """
        Parameters:
            visualization_type: "arrow" or "triangles"
        """

        # set up value range
        allvals = sum(
            [list(av.values()) for s, av in state_action_map.items()],
            [])
        absvals = [abs(v) for v in allvals]
        absvmax = max(absvals)
        if value_range is None:
            value_range = [-absvmax, absvmax]
        else:
            absvmax = max([abs(v) for v in value_range])
        vmin, vmax = value_range

        if isinstance(color_value_func, str):
            colorrange = plt.get_cmap(color_value_func)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            color_value_map = cmx.ScalarMappable(norm=color_norm,
                                                 cmap=colorrange)
            color_value_func = lambda v: color_value_map.to_rgba(v)

        # format mapping for plotting
        if isinstance(next(iter(state_action_map)), (dict, frozendict)):
            to_plot = {}
            for s, a_v in state_action_map.items():
                if self.gw.is_absorbing(s):
                    continue
                if (not plot_over_walls) and (s in self.gw.walls):
                    continue
                s_ = (s['x'], s['y'])
                to_plot[s_] = {}
                for a, v in a_v.items():
                    a_ = (a.get('dx', 0), a.get('dy', 0))
                    to_plot[s_][a_] = v
        elif isinstance(next(iter(state_action_map)), (tuple, list)):
            to_plot = {}
            for s, a_v in state_action_map.items():
                if self.gw.is_absorbing(s):
                    continue
                if (not plot_over_walls) and (s in self.gw.walls):
                    continue
                to_plot[s] = {**a_v}
        else:
            raise Exception("unknown state representation")

        def plot_state_action_map_as_arrows():
            for s, av in to_plot.items():
                x, y = s
                for a, v in av.items():
                    dx, dy = a
                    arrowColor = color_value_func(v)
                    mag = abs(v) / absvmax
                    mag *= .5
                    if (dx != 0) or (dy != 0):
                        patch = Arrow(x + .5, y + .5, dx * mag, dy * mag,
                                      width=arrow_width,
                                      color=arrowColor)
                    else:
                        patch = Circle((x + .5, y + .5), radius=mag * .9,
                                       fill=False, color=arrowColor)
                    self.ax.add_patch(patch)

        def plot_state_action_map_as_triangles():
            sav_params = []
            for (x, y), a_v in to_plot.items():
                for (dx, dy), v in a_v.items():
                    vertices = {
                        (0, 0): [(.3, .3), (.7, .3), (.7, .7), (.3, .7)],
                        (-1, 0): [(.5, .5), (0, 0), (0, 1)],
                        (1, 0): [(.5, .5), (1, 0), (1, 1)],
                        (0, 1): [(.5, .5), (0, 1), (1, 1)],
                        (0, -1): [(.5, .5), (0, 0), (1, 0)],
                    }[(dx, dy)]
                    vertices = [(x + ix, y + iy) for ix, iy in vertices]
                    av_params = list(zip(*vertices)) + [
                        colors.to_hex(color_value_func(v))]
                    if (dx, dy) == (0, 0):
                        sav_params.extend(av_params)
                    else:
                        sav_params = av_params + sav_params
            _ = self.ax.fill(*sav_params)

        def plot_state_action_map_numbers():
            for (x, y), a_v in to_plot.items():
                for (dx, dy), v in a_v.items():
                    ann_params = {
                        (0, 0): {"xy": (.5, .5), "ha": "center",
                                 "va": "center"},
                        (-1, 0): {"xy": (.05, .5), "ha": "left",
                                  "va": "center"},
                        (1, 0): {"xy": (.95, .5), "ha": "right",
                                 "va": "center"},
                        (0, 1): {"xy": (.5, .95), "ha": "center", "va": "top"},
                        (0, -1): {"xy": (.5, .05), "ha": "center",
                                  "va": "bottom"}
                    }[(dx, dy)]
                    ann_params['xy'] = (
                    ann_params['xy'][0] + x, ann_params['xy'][1] + y)
                    contrast_color = get_contrast_color(color_value_func(v))
                    contrast_color = contrast_color if contrast_color == 'white' else 'black'
                    self.ax.annotate(text=f"{v:+.1f}",
                                     **{**dict(color=contrast_color),
                                        **numbers_kw, **ann_params})

        if "arrow" in visualization_type:
            plot_state_action_map_as_arrows()
        elif "triangle" in visualization_type:
            plot_state_action_map_as_triangles()
        else:
            raise ValueError("Unknown visualization type")
        if show_numbers:
            if numbers_kw is None:
                numbers_kw = dict(fontsize=10)
            if "arrow" in visualization_type:
                numbers_kw['color'] = "k"
            plot_state_action_map_numbers()
        return self

    def plot_policy(
        self,
        policy: Union[TabularPolicy, dict],
        plot_over_walls=False
    ) -> "GridWorldPlotter":
        # if isinstance(policy, TabularPolicy):
        #     policy = policy.policy_dict
        return self.plot_state_action_map(
            state_action_map=policy,
            plot_over_walls=plot_over_walls,
            value_range=[0, 1],
            color_value_func=lambda v: 'k'
        )

    def title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)
        return self

    def annotate(
        self,
        s,
        a=None,
        text="",
        outlinewidth=0,
        outlinecolor='black',
        fontsize=10,
        ha='center',
        va='center',
        **kwargs
    ):
        kwargs = {
            'fontsize': fontsize,
            'ha': ha,
            'va': va,
            **kwargs
        }

        if isinstance(s, (tuple, list)):
            s = s
        elif isinstance(s, (dict, frozendict)):
            s = (s['x'], s['y'])
        text = self.ax.text(s[0] + .5, s[1] + .5, text, **kwargs)
        if outlinewidth > 0:
            text.set_path_effects([
                path_effects.Stroke(linewidth=outlinewidth,
                                    foreground=outlinecolor),
                path_effects.Normal()
            ])
        return self

    #shortcuts
    def pSA(self, *args, **kwargs):
        return self.plot_state_action_map(*args, **kwargs)

    def pS(self, *args, **kwargs):
        return self.plot_state_map(*args, **kwargs)

    def pT(self, *args, **kwargs):
        return self.plot_trajectory(*args, **kwargs)

    def pPi(self, *args, **kwargs):
        return self.plot_policy(*args, **kwargs)

    def t(self, *args, **kwargs):
        return self.title(*args, **kwargs)
