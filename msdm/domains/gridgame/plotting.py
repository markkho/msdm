import numpy as np
from functools import reduce
from typing import Mapping, Union, Callable, Hashable
from numbers import Number
from functools import reduce
from tqdm import tqdm 

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patheffects as path_effects
from matplotlib import animation 

from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy
from msdm.domains.gridgame.tabulargridgame import TabularGridGame
from msdm.core.assignment.assignmentmap import AssignmentMap



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

AGENT_COLORS = ["indigo","deepskyblue","magenta"]




def get_contrast_color(color):
    r, g, b = colors.to_rgb(color)
    luminance = (0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2) ** .5
    if luminance < .7:
        return "white"
    return 'grey'


class GridGamePlotter:
    def __init__(self, gg: TabularGridGame, ax: plt.Axes):
        self.gg = gg
        self.ax = ax
        self.ax.axis('off')
        self.ax.set_xlim(-0.1, self.gg.width + .1)
        self.ax.set_ylim(-0.1, self.gg.height + .1)
        self.ax.axis('equal')
        # Whenever an agent is plotted, will be assigned a color stored here. Use
        # the associated color for other plotting for consistency 
        self.agent_colors = AssignmentMap()
        self.agents = AssignmentMap()

    def plot_features(self, featurecolors, edgecolor='darkgrey') -> "GridGamePlotter":
        """Plot gridgame features"""
        ss = self.gg.state_list
        for x in range(self.gg.width):
            for y in range(self.gg.height):
                feature = self.check_features(x,y)
                xy = (x, y)
                color = featurecolors.get(feature["type"], 'w')
                square = Rectangle(xy, 1, 1,
                                   facecolor=color,
                                   edgecolor=edgecolor,
                                   linewidth=2)
                self.ax.add_patch(square)
        return self
    
    def check_features(self,x,y):
        current_features = []
        for obstacle in self.gg.obstacles: 
            if obstacle["x"] == x and obstacle["y"] == y:
                current_features.append(obstacle)
        # default if no relevant features are found 
        if len(current_features) == 0:
            return {"type":"background"}
        if len(current_features) > 1:
            print("Overlapping features. Choosing first one in list for visualization")
            return current_features[0]
        else:
            return current_features[0]

    def plot_outer_box(self):
        outerbox = Rectangle((0, 0), self.gg.width, self.gg.height,
                             fill=False, edgecolor='black',
                             linewidth=2)
        self.ax.add_patch(outerbox)
        return self

    def plot_walls(self, facecolor='k', edgecolor='darkgrey'):
        for ws in self.gg.walls:
            x_coords = (ws["start"]["x"], ws["end"]["x"])
            y_coords = (ws["start"]["y"],ws["end"]["y"])
            # Left wall case 
            if x_coords[0] > x_coords[1]:
                x_coords = (x_coords[0],x_coords[0])
                y_coords = (y_coords[0],y_coords[0]+1)
                self.ax.plot(x_coords,y_coords,color="cyan",linewidth=2)
            # Right wall case 
            elif x_coords[0] < x_coords[1]:
                x_coords = (x_coords[1],x_coords[1])
                y_coords = (y_coords[0],y_coords[0]+1)
                self.ax.plot(x_coords,y_coords,color="cyan",linewidth=2)
            # Lower wall case 
            elif y_coords[0] > y_coords[1]: 
                x_coords = (x_coords[0],x_coords[0]+1)
                y_coords = (y_coords[0],y_coords[0])
                self.ax.plot(x_coords,y_coords,color="cyan",linewidth=2)
            # Upper wall case
            elif y_coords[0] < y_coords[1]:
                x_coords = (x_coords[0],x_coords[0]+1)
                y_coords = (y_coords[1],y_coords[1])
                self.ax.plot(x_coords,y_coords,color="cyan",linewidth=2)
            else:
                self.ax.plot(x_coords,y_coords,color="cyan",linewidth=2)
        return self
    
    def plot_fences(self):
        for fence in self.gg.fences:
            x_coords = (fence["start"]["x"], fence["end"]["x"])
            y_coords = (fence["start"]["y"],fence["end"]["y"])
            # Left wall case 
            if x_coords[0] > x_coords[1]:
                x_coords = (x_coords[0],x_coords[0])
                y_coords = (y_coords[0],y_coords[0]+1)
                self.ax.plot(x_coords,y_coords,color="green",linewidth=2)
            # Right wall case 
            elif x_coords[0] < x_coords[1]:
                x_coords = (x_coords[1],x_coords[1])
                y_coords = (y_coords[0],y_coords[0]+1)
                self.ax.plot(x_coords,y_coords,color="green",linewidth=2)
            # Lower wall case 
            elif y_coords[0] > y_coords[1]: 
                x_coords = (x_coords[0],x_coords[0]+1)
                y_coords = (y_coords[0],y_coords[0])
                self.ax.plot(x_coords,y_coords,color="green",linewidth=2)
            # Upper wall case
            elif y_coords[0] < y_coords[1]:
                x_coords = (x_coords[0],x_coords[0]+1)
                y_coords = (y_coords[1],y_coords[1])
                self.ax.plot(x_coords,y_coords,color="green",linewidth=2)
        return self
        


    def plot_initial_states(self, markersize=15, featurecolors=None):
        for i,agent in enumerate(self.gg.agents):
            self.agent_colors[agent["name"]] = AGENT_COLORS[i]
            x, y = agent['x'], agent['y']
            self.agents[agent["name"]] = self.ax.plot(x + .5, y + .5,
                         markeredgecolor=self.agent_colors[agent["name"]],
                         marker="o",
                         markersize=markersize,
                         markeredgewidth=2,
                         fillstyle='none')[0]
        return self
    
    def plot_new_state(self,state):
        for i,agent in enumerate(self.gg.agents):
            x,y = state[agent["name"]]["x"], state[agent["name"]]["y"]
            self.agents[agent["name"]].set_data(x+.5,y+.5)
        return self

    def plot_absorbing_states(self, markersize=30, featurecolors=None):
        for i,goal in enumerate(self.gg.goals):
            # sdict = dict(zip([v.name for v in s.variables], s.values))
            # x, y = sdict['x'], sdict['y']
            x, y = goal['x'], goal['y']
            if len(goal["owners"]) == 1:
                marker_color = self.agent_colors[goal["owners"][0]]
            else:
                marker_color = "cornflowerblue"
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor=marker_color,
                         markerfacecolor=marker_color,
                         marker='*',
                         fillstyle="full",
                         markersize=markersize,
                         markeredgewidth=2)

    def plot_trajectory(self,
                        stateTraj,
                        agent_list,
                        actionTraj=None,  # not implemented yet
                        color='k',
                        outline=False,
                        outlineColor='w',
                        jitterMean=0,
                        jitterVar=.1,
                        endJitter=False,
                        linewidth=1,
                        **kwargs) -> "GridGamePlotter":
        if actionTraj is not None:
            assert len(stateTraj) == len(actionTraj)

        xys = {agent:[] for agent in agent_list}
        for s in stateTraj:
            if self.gg.is_terminal(s):
                break
            for agent in agent_list:
                xys[agent].append((s[agent]["x"],s[agent]["y"]))
        for agent in agent_list:
            if len(xys[agent]) == 2:
                p0 = tuple(np.array(xys[agent][0]) + .5)
                p2 = tuple(np.array(xys[agent][1]) + .5)
                p1 = np.array([(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]) \
                     + np.random.normal(0, jitterVar, 2)
                if endJitter:
                    p0 = tuple(
                        np.array(p0) + np.random.normal(jitterMean, jitterVar, 2))
                    p1 = tuple(
                        np.array(p1) + np.random.normal(jitterMean, jitterVar, 2))
                segments = [[p0, p1, p2], ]
            elif (len(xys[agent]) == 3) and (xys[agent][0] == xys[agent][2]):
                p0 = tuple(np.array(xys[agent][0]) + .5)
                p2 = tuple(np.array(xys[agent][1]) + .5)
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
                for xy in xys[agent]:
                    jitter = np.random.normal(jitterMean, jitterVar, 2)
                    coord = np.array(xy) + .5 + jitter
                    state_coords.append(tuple(coord))
                if not endJitter:
                    state_coords[0] = tuple(np.array(xys[agent][0]) + .5)
                    state_coords[-1] = tuple(np.array(xys[agent][-1]) + .5)
                join_point = state_coords[0]
                segments = []
                for i, xy in enumerate(state_coords[:-1]):
                    nxy = state_coords[i + 1]

                    segment = []
                    segment.append(join_point)
                    segment.append(xy)
                    if i < len(xys[agent]) - 2:
                        join_point = tuple(np.mean([xy, nxy], axis=0))
                        segment.append(join_point)
                    else:
                        segment.append(nxy)
                    segments.append(segment)

            outline_patches = []
            if outline:
                for segment, step in zip(segments, xys[agent][:-1]):
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    path = Path(segment, codes)
                    outline_patch = patches.PathPatch(path, facecolor='none',
                                                      capstyle='butt',
                                                      edgecolor=outlineColor,
                                                      linewidth=linewidth * 2)
                    self.ax.add_patch(outline_patch)
                    outline_patches.append(outline_patch)

            xys_patches = []
            for segment, step in zip(segments, xys[agent][:-1]):
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(segment, codes)

                patch = patches.PathPatch(path, facecolor='none', capstyle='butt',
                                          edgecolor=color, linewidth=linewidth,
                                          **kwargs)
                xys_patches.append(patch)
                self.ax.add_patch(patch)
        return self

    def plot_state_map(self,
                       positionMap: Mapping,
                       plotOverWalls=False,
                       fontsize=10,
                       showNumbers=True,
                       valueRange=None,
                       showColors=True,
                       isCategorical=False,
                       colorValueFunc="bwr_r") -> "GridGamePlotter":
        if len(positionMap) == 0:
            return self
        # state map - colors / numbers
        vmax_abs = max(abs(v) for k, v in positionMap.items())
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
        for s, v in positionMap.items():
            if (not plotOverWalls) and (s in [(wall["x"],wall["y"]) for wall in self.gg.walls]):
                continue
            if isinstance(s, tuple) or isinstance(s, list):
                xy = s
            else:
                raise Exception("unknown state representation")

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
                             color=get_contrast_color(color),
                             horizontalalignment='center',
                             verticalalignment='center')
        return self

    def plot_state_action_map(self,
                              stateActionMap: Mapping[
                               Hashable, Mapping[Hashable, Number]],
                              plotOverWalls=False,
                              valueRange=None,
                              colorvalue_func: Union[Callable, str]="bwr_r",
                              arrowWidth=.1) -> "GridGamePlotter":
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
            if (not plotOverWalls) and (s in self.gg.walls):
                continue

            if isinstance(s, tuple) or isinstance(s, list):
                x, y = s
            else:
                raise Exception("unknown state representation")

            for a, v in av.items():
                dx, dy = a.get('x', 0.0), a.get('y', 0.0)
                arrowColor = colorvalue_func(v)
                mag = abs(v) / absvmax
                mag *= .5
                if (dx != 0) or (dy != 0):
                    patch = Arrow(x + .5, y + .5, dx * mag, dy * mag,
                                  width=arrowWidth,
                                  color=arrowColor)
                else:
                    patch = Circle((x + .5, y + .5), radius=mag * .9,
                                   fill=False, color=arrowColor)
                self.ax.add_patch(patch)
        return self
    
    def plot_weights(self,
                       weightMap: Mapping,
                       plotOverWalls=False,
                       fontsize=10,
                       showNumbers=True,
                       valueRange=None,
                       showColors=True,
                       isCategorical=False,
                       colorValueFunc="bwr_r") -> "GridGamePlotter":
        
        if len(weightMap) == 0:
            return self
        # state map - colors / numbers
        vmax_abs = max(abs(v) for k, v in weightMap.items())
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
            
        for s, v in weightMap.items():
            if (not plotOverWalls) and (s in [(wall["x"],wall["y"]) for wall in self.gg.walls]):
                continue
            if isinstance(s, tuple) or isinstance(s, list):
                xy = s
            else:
                raise Exception("unknown state representation")

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
                             color=get_contrast_color(color),
                             horizontalalignment='center',
                             verticalalignment='center')
        return self

        
    def plot_policy(self, policy: TabularMultiAgentPolicy) -> "GridGamePlotter":
        
        for agent in policy.single_agent_policies:
            continue 
        return self.plot_state_action_map(
            stateActionMap=policy,
            plotOverWalls=False,
            valueRange=[0, 1],
            colorvalue_func=lambda v: 'k'
        )
    
    def title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)
        return self

    def annotate(self, s, a=None, text="",
                 outlinewidth=0, outlinecolor='black',
      fontsize=10, ha='center', va='center', **kwargs):
        kwargs = {
            'fontsize': fontsize,
            'ha': ha,
            'va': va,
            **kwargs
        }
        text = self.ax.text(s['x'] + .5, s['y'] + .5, text, **kwargs)
        if outlinewidth > 0:
            text.set_path_effects([
                path_effects.Stroke(linewidth=outlinewidth,
                                    foreground=outlinecolor),
                path_effects.Normal()
            ])

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
