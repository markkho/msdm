import numpy as np
from typing import Mapping, Union, Callable
from numbers import Number

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx

from pyrlap.pyrlap2.core import State, TERMINALSTATE, Action, TabularPolicy
from pyrlap.pyrlap2.domains.gridworld.mdp import GridWorld

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


def getContrastColor(color):
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
        self.ax.axis('scaled')

    def plotFeatures(self, featureColors, edgecolor='darkgrey') -> "GridWorldPlotter":
        """Plot gridworld features"""
        ss = self.gw.states
        for s in ss:
            if s == TERMINALSTATE:
                continue
            sdict = dict(zip([v.name for v in s.variables], s.values))
            xy = (sdict['x'], sdict['y'])
            f = self.gw.locationFeatures.get(s)
            color = featureColors.get(f, 'w')
            square = Rectangle(xy, 1, 1,
                               facecolor=color,
                               edgecolor=edgecolor,
                               linewidth=2)
            self.ax.add_patch(square)
        outerbox = Rectangle((0, 0), self.gw.width, self.gw.height,
                             fill=False, edgecolor='black',
                             linewidth=2)
        self.ax.add_patch(outerbox)
        return self

    def plotInitStates(self, markerSize=15):
        for s in self.gw.initStates:
            sdict = dict(zip([v.name for v in s.variables], s.values))
            x, y = sdict['x'], sdict['y']
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor='cornflowerblue',
                         marker='o',
                         markersize=markerSize,
                         markeredgewidth=2,
                         fillstyle='none')
        return self

    def plotAbsorbingStates(self, markerSize=15):
        for s in self.gw.absorbingStates:
            sdict = dict(zip([v.name for v in s.variables], s.values))
            x, y = sdict['x'], sdict['y']
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor='cornflowerblue',
                         marker='x',
                         markersize=markerSize,
                         markeredgewidth=2)

    def plotTrajectory(self,
                       stateTraj,
                       actionTraj=None,  # not implemented yet
                       color='k',
                       outline=False,
                       outlineColor='w',
                       jitterMean=0,
                       jitterVar=.1,
                       endJitter=False,
                       linewidth=1,
                       **kwargs) -> "GridWorldPlotter":
        if actionTraj is not None:
            assert len(stateTraj) == len(actionTraj)

        xys = []
        for s in stateTraj:
            if s == TERMINALSTATE:
                break
            s = dict(zip([v.name for v in s.variables], s.values))
            xys.append((s['x'], s['y']))

        if len(xys) == 2:
            p0 = tuple(np.array(xys[0]) + .5)
            p2 = tuple(np.array(xys[1]) + .5)
            p1 = np.array([(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]) \
                 + np.random.normal(0, jitterVar, 2)
            if endJitter:
                p0 = tuple(
                    np.array(p0) + np.random.normal(jitterMean, jitterVar, 2))
                p1 = tuple(
                    np.array(p1) + np.random.normal(jitterMean, jitterVar, 2))
            segments = [[p0, p1, p2], ]
        elif (len(xys) == 3) and (xys[0] == xys[2]):
            p0 = tuple(np.array(xys[0]) + .5)
            p2 = tuple(np.array(xys[1]) + .5)
            if abs(p0[0] - p2[0]) > 0:  # horizontal
                jitter = np.array(
                    [0, np.random.normal(jitterMean, jitterVar * 2)])
                p2 = p2 - np.array([.25, 0])
            else:  # vertical
                jitter = np.array(
                    [np.random.normal(jitterMean, jitterVar * 2), 0])
                p2 = p2 - np.array([0, .25])
            p1 = p2 + jitter
            p3 = p2 - jitter
            segments = [[p0, p1, p2], [p2, p3, p0]]
        else:
            state_coords = []
            for xy in xys:
                jitter = np.random.normal(jitterMean, jitterVar, 2)
                coord = np.array(xy) + .5 + jitter
                state_coords.append(tuple(coord))
            if not endJitter:
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
                                                  edgecolor=outlineColor,
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

    def plotStateMap(self,
                     stateMap: Mapping[State, Number],
                     plotOverWalls=False,
                     fontsize=10,
                     showNumbers=True,
                     valueRange=None,
                     showColors=True,
                     isCategorical=False,
                     colorValueFunc="bwr_r") -> "GridWorldPlotter":
        if len(stateMap) == 0:
            return self
        # state map - colors / numbers
        vmax_abs = max(abs(v) for k, v in stateMap.items())
        if valueRange is None:
            valueRange = [-vmax_abs, vmax_abs]
        vmin, vmax = valueRange
        if isCategorical:
            colorValueFunc = lambda i: DISTINCT_COLORS[
                int(i) % len(DISTINCT_COLORS)]
        elif isinstance(colorValueFunc, str):
            colorrange = plt.get_cmap(colorValueFunc)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            colorvalue_map = cmx.ScalarMappable(norm=color_norm,
                                                cmap=colorrange)
            colorValueFunc = lambda v: colorvalue_map.to_rgba(v)
        for s, v in stateMap.items():
            assert isinstance(s, State)
            if s == TERMINALSTATE:
                continue
            if (not plotOverWalls) and (s in self.gw.walls):
                continue
            sdict = dict(zip([v.name for v in s.variables], s.values))
            xy = (sdict['x'], sdict['y'])
            color = 'w'
            if showColors:
                color = colorValueFunc(v)
                square = Rectangle(xy, 1, 1,
                                   color=color,
                                   ec='k', lw=2)
                self.ax.add_patch(square)
            if showNumbers:
                self.ax.text(xy[0] + .5, xy[1] + .5,
                             f"{v : .2f}",
                             fontsize=fontsize,
                             color=getContrastColor(color),
                             horizontalalignment='center',
                             verticalalignment='center')
        return self

    def plotStateActionMap(self,
                           stateActionMap: Mapping[
                               State, Mapping[Action, Number]],
                           plotOverWalls=False,
                           valueRange=None,
                           colorvalue_func: Union[Callable, str]="bwr_r",
                           arrowWidth=.1) -> "GridWorldPlotter":
        allvals = sum([list(av.values()) for s, av in stateActionMap.items()],
                      [])
        absvals = [abs(v) for v in allvals]
        absvmax = max(absvals)
        if valueRange is None:
            valueRange = [-absvmax, absvmax]
        else:
            absvmax = max([abs(v) for v in valueRange])
        vmin, vmax = valueRange

        if isinstance(colorvalue_func, str):
            colorrange = plt.get_cmap(colorvalue_func)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            colorvalue_map = cmx.ScalarMappable(norm=color_norm,
                                                cmap=colorrange)
            colorvalue_func = lambda v: colorvalue_map.to_rgba(v)

        for s, av in stateActionMap.items():
            if s == TERMINALSTATE:
                continue
            if (not plotOverWalls) and (s in self.gw.walls):
                continue
            sdict = dict(zip([v.name for v in s.variables], s.values))
            x, y = sdict['x'], sdict['y']
            for a, v in av.items():
                adict = dict(zip([v.name for v in a.variables], a.values))
                dx, dy = adict['ax'], adict['ay']
                arrowColor = colorvalue_func(v)
                mag = abs(v) / absvmax
                mag *= .5
                if (dx != 0) or (dy != 0):
                    patch = Arrow(x + .5, y + .5, dx * mag, dy * mag,
                                  width=arrowWidth,
                                  color=arrowColor)
                else:
                    patch = Circle((x + .5, y + .5), radius=mag * .9,
                                   fill=False)
                self.ax.add_patch(patch)
        return self

    def plotPolicy(self, policy: Union[TabularPolicy, dict]) -> "GridWorldPlotter":
        if isinstance(policy, TabularPolicy):
            policy = policy.policydict
        return self.plotStateActionMap(
            stateActionMap=policy,
            plotOverWalls=False,
            valueRange=[0, 1],
            colorvalue_func=lambda v: 'k'
        )

    def title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)
        return self
