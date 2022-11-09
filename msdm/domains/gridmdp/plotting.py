import numpy as np
from typing import Mapping, Union, Callable, Hashable
from numbers import Number

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patheffects as path_effects

from msdm.domains.gridmdp import GridMDP, Location, GridAction
from msdm.core.mdp.policy import Policy
from msdm.core.distributions import FiniteDistribution

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

class GridMDPPlotter:
    def __init__(self, grid_mdp: GridMDP, ax: plt.Axes):
        self.grid_mdp = grid_mdp
        self.ax = ax
        self.ax.axis('off')
        self.ax.set_xlim(-0.1, self.grid_mdp.width + .1)
        self.ax.set_ylim(-0.1, self.grid_mdp.height + .1)
        ax.set_aspect('equal')

    def fill_features(
        self,
        feature_colors,
        default_color=None,
        Rectangle_kwargs=None
    ):
        artists = []
        for feature in self.grid_mdp.feature_list:
            if feature in feature_colors:
                rects = self.fill_feature(feature, feature_colors[feature], Rectangle_kwargs)
                artists.extend(rects)
            elif default_color is not None:
                rects = self.fill_feature(feature, default_color, Rectangle_kwargs)
                artists.extend(rects)
        return artists

    def mark_features(
        self,
        feature_markers,
        plot_kwargs=None
    ):
        markers = []
        for feature, marker in feature_markers.items():
            m = self.mark_feature(feature, marker, plot_kwargs)
            markers.append(m)
        return markers

    def mark_location(
        self,
        location,
        marker,
        plot_kwargs=None
    ):
        if plot_kwargs is None:
            plot_kwargs = {}
        x, y = self.as_xy(location)
        marker = self.ax.plot(
            x + .5,
            y + .5,
            marker=marker,
            **plot_kwargs
        )
        return marker

    def plot_location_trajectory(
        self,
        location_trajectory,
        outline=False,
        jitter_mean=0,
        jitter_var=0,
        end_jitter=False,
        main_PathPatch_kwargs=None,
        outline_PathPatch_kwargs=None
    ):
        if main_PathPatch_kwargs is None:
            main_PathPatch_kwargs = {}
        main_PathPatch_kwargs = {
            **dict(
                facecolor='none',
                capstyle='butt',
                edgecolor='k',
                linewidth=1
            ),
            **main_PathPatch_kwargs,
        }

        if outline_PathPatch_kwargs is None:
            outline_PathPatch_kwargs = {}
        outline_PathPatch_kwargs = {
            **dict(
                facecolor='none',
                capstyle='butt',
                edgecolor='w',
                linewidth=main_PathPatch_kwargs['linewidth']*2
            ),
            **outline_PathPatch_kwargs,
        }

        xys = [self.as_xy(loc) for loc in location_trajectory]
        spline_segments = self._calculate_spline_segments(xys, jitter_mean, jitter_var, end_jitter)

        artists = []
        if outline:
            outline_splines = \
                self._plot_spline_segments(spline_segments, outline_PathPatch_kwargs)
            artists.extend(outline_splines)
        main_splines = \
            self._plot_spline_segments(spline_segments, main_PathPatch_kwargs)
        artists.extend(main_splines)
        return artists

    def plot_location_map(
        self,
        location_map,
        fontsize=10,
        show_numbers=True,
        vmin=None,
        vmax=None,
        show_colors=True,
        is_categorical=False,
        color_value_func="bwr_r"
    ):
        vmax_abs = self._max_absolute_value(location_map)
        if vmin is None:
            vmin = -vmax_abs
        if vmax is None:
            vmax = vmax_abs

        if is_categorical:
            color_value_func = lambda i: DISTINCT_COLORS[
                int(i) % len(DISTINCT_COLORS)]

        color_value_func = self._get_color_value_func(color_value_func, vmin, vmax)

        artists = []
        for loc, v in location_map.items():
            x, y = self.as_xy(loc)
            color = 'w'
            if show_colors:
                color = color_value_func(v)
                square = Rectangle(
                    (x, y), 1, 1,
                    color=color,
                    ec='k', lw=2)
                self.ax.add_patch(square)
                artists.append(square)
            if show_numbers:
                number = self.ax.text(
                    x + .5, y + .5,
                    f"{v : .2f}",
                    fontsize=fontsize,
                    color=get_contrast_color(color),
                    horizontalalignment='center',
                    verticalalignment='center'
                )
                artists.append(number)
        return artists
    
    def plot_policy(
        self,
        policy : Policy,
        **kws
    ):
        return self.plot_location_action_map(
            policy,
            vmin=0.,
            vmax=1.0,
            color_value_func=lambda loc : 'k',
            **kws
        )

    def plot_location_action_map(
        self,
        location_action_map: Mapping,
        vmin=None,
        vmax=None,
        color_value_func: Union[Callable, str] = "bwr_r",
        arrow_width=.1,
        show_numbers=False,
        numbers_kw=None,
        visualization_type="arrow"
    ):
        """
        Parameters:
            visualization_type: str
                This can be "arrow" or "triangles"
        """
        max_abs_val = self._max_absolute_value(location_action_map)
        if vmin is None:
            vmin = -max_abs_val
        else:
            max_abs_val = abs(vmin)
        if vmax is None:
            vmax = max_abs_val
        else:
            max_abs_val = abs(vmax)
        color_value_func = self._get_color_value_func(color_value_func, vmin, vmax)

        # format mapping for plotting
        xy_dxdy_values = {}
        for loc, a_v in location_action_map.items():
            xy = self.as_xy(loc)
            xy_dxdy_values[xy] = {}
            for a, v in a_v.items():
                dxdy = self.as_dxdy(a)
                xy_dxdy_values[xy][dxdy] = v

        if "arrow" in visualization_type:
            artists = self._plot_location_action_map_as_arrows(xy_dxdy_values, color_value_func, arrow_width, max_abs_val)
        elif "triangle" in visualization_type:
            artists = self._plot_location_action_map_as_triangles(xy_dxdy_values, color_value_func)
        else:
            raise ValueError("Unknown visualization type")
        if show_numbers:
            if numbers_kw is None:
                numbers_kw = dict(fontsize=10)
            if "arrow" in visualization_type:
                numbers_kw['color'] = "k"
            numbers = self._plot_location_action_map_numbers(xy_dxdy_values, color_value_func, numbers_kw)
            artists.extend(numbers)
        return artists

    def plot_outer_box(self):
        outerbox = Rectangle(
            (0, 0), self.grid_mdp.width, self.grid_mdp.height,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(outerbox)
        return [outerbox, ]

    def fill_feature(
        self,
        feature,
        color,
        Rectangle_kwargs=None
    ):
        if Rectangle_kwargs is None:
            Rectangle_kwargs = {}
        Rectangle_kwargs = {
            **dict(
                edgecolor='darkgrey',
                linewidth=2
            ),
            **Rectangle_kwargs
        }
        artists = []
        for loc in self.grid_mdp.locations_with(feature):
            x, y = self.as_xy(loc)
            square = Rectangle(
                (x, y), 1, 1,
                facecolor=color,
                **Rectangle_kwargs
            )
            art = self.ax.add_patch(square)
            artists.append(art)
        return artists

    def mark_feature(
        self,
        feature,
        marker,
        plot_kwargs=None
    ):
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs = {
            **dict(
                markeredgecolor='cornflowerblue',
                markersize=15,
                markeredgewidth=2,
                fillstyle='none'
            ),
            **plot_kwargs
        }
        artists = []
        for loc in self.grid_mdp.locations_with(feature):
            x, y = self.as_xy(loc)
            artist = self.ax.plot(
                x + .5,
                y + .5,
                marker=marker,
                **plot_kwargs
            )
            artists.append(artist)
        return artists

    def _plot_spline_segments(
        self,
        spline_segments,
        PathPatch_kwargs
    ):
        xys_patches = []
        for segment in spline_segments:
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(segment, codes)
            patch = patches.PathPatch(
                path,
                **PathPatch_kwargs
            )
            xys_patches.append(patch)
            self.ax.add_patch(patch)
        return xys_patches

    def _2spline(self, xys, jitter_mean, jitter_var, end_jitter):
        p0 = tuple(np.array(xys[0]) + .5)
        p2 = tuple(np.array(xys[1]) + .5)
        p1 = np.array([(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]) \
             + np.random.normal(0, jitter_var, 2)
        if end_jitter:
            p0 = tuple(
                np.array(p0) + np.random.normal(jitter_mean, jitter_var, 2))
            p1 = tuple(
                np.array(p1) + np.random.normal(jitter_mean, jitter_var, 2))
        spline_segments = [[p0, p1, p2], ]
        return spline_segments

    def _self_loop_spline(self, xys, jitter_mean, jitter_var):
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
        spline_segments = [[p0, p1, p2], [p2, p3, p0]]
        return spline_segments

    def _long_spline(self, xys, jitter_mean, jitter_var, end_jitter):
        state_coords = []
        for xy in xys:
            jitter = np.random.normal(jitter_mean, jitter_var, 2)
            coord = np.array(xy) + .5 + jitter
            state_coords.append(tuple(coord))
        if not end_jitter:
            state_coords[0] = tuple(np.array(xys[0]) + .5)
            state_coords[-1] = tuple(np.array(xys[-1]) + .5)
        join_point = state_coords[0]
        spline_segments = []
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
            spline_segments.append(segment)
        return spline_segments

    def _calculate_spline_segments(self, xys, jitter_mean, jitter_var, end_jitter):
        is_self_loop = (len(xys) == 3) and (xys[0] == xys[2])
        if len(xys) == 2:
            spline_segments = self._2spline(xys, jitter_mean, jitter_var, end_jitter)
        elif is_self_loop:
            spline_segments = self._self_loop_spline(xys, jitter_mean, jitter_var)
        else:
            spline_segments = self._long_spline(xys, jitter_mean, jitter_var, end_jitter)
        return spline_segments

    def _plot_location_action_map_as_triangles(self, xy_dxdy_values, color_value_func):
        sav_params = []
        for (x, y), dxdy_v in xy_dxdy_values.items():
            for (dx, dy), v in dxdy_v.items():
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
        artists = self.ax.fill(*sav_params)
        return artists

    def _plot_location_action_map_as_arrows(
        self,
        xy_dxdy_values,
        color_value_func,
        arrow_width,
        max_abs_val
    ):
        artists = []
        for (x, y), dxdy_v in xy_dxdy_values.items():
            for (dx, dy), v in dxdy_v.items():
                arrowColor = color_value_func(v)
                mag = abs(v) / max_abs_val
                mag *= .5
                if (dx != 0) or (dy != 0):
                    patch = Arrow(x + .5, y + .5, dx * mag, dy * mag,
                                  width=arrow_width,
                                  color=arrowColor)
                else:
                    patch = Circle((x + .5, y + .5), radius=mag * .9,
                                   fill=False, color=arrowColor)
                self.ax.add_patch(patch)
                artists.append(patch)
        return artists

    def _plot_location_action_map_numbers(self, xy_dxdy_values, color_value_func, numbers_kw):
        artists = []
        for (x, y), dxdy_v in xy_dxdy_values.items():
            for (dx, dy), v in dxdy_v.items():
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
                artist = self.ax.annotate(
                    text=f"{v:+.1f}",
                    **{**dict(color=contrast_color),
                    **numbers_kw, **ann_params}
                )
                artists.append(artist)
        return artists

    def _get_color_value_func(self, color_value_func, vmin, vmax):
        if isinstance(color_value_func, str):
            colorrange = plt.get_cmap(color_value_func)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            color_value_map = cmx.ScalarMappable(norm=color_norm,
                                                 cmap=colorrange)
            color_value_func = lambda v: color_value_map.to_rgba(v)
        elif isinstance(color_value_func, Callable):
            pass
        else:
            raise Exception("Unrecognized color value function")
        return color_value_func

    def _max_absolute_value(self, collection):
        max_abs_val = -float('inf')
        for ele in collection:
            if isinstance(collection, dict):
                ele = collection[ele]
            if isinstance(ele, (dict, FiniteDistribution)):
                for v in ele.values():
                    max_abs_val = max(max_abs_val, abs(v))
            elif isinstance(ele, (tuple, list)):
                for v in ele:
                    max_abs_val = max(max_abs_val, abs(v))
            elif isinstance(ele, (int, float)):
                v = ele
                max_abs_val = max(max_abs_val, abs(v))
            else:
                raise Exception("Unknown type")
        return max_abs_val

    def as_dxdy(self, action):
        if isinstance(action, dict):
            return action['dx'], action['dy']
        elif isinstance(action, (tuple, list)):
            return action[0], action[1]
        else:
            raise Exception("Unrecognized action representation")

    def as_xy(self, loc):
        if isinstance(loc, dict):
            return loc['x'], loc['y']
        elif isinstance(loc, (tuple, list)):
            return loc[0], loc[1]
        else:
            raise Exception("Unrecognized location representation")
