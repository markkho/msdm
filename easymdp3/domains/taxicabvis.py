from itertools import product

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.animation as animation

from easymdp3.domains.gridworldvis \
    import visualize_states as visualize_locs, \
    visualize_walls


def plot_circle(state, ax=None, r=.15,
                sub_state=(1, 1),
                sub_rows=3,
                sub_cols=3,
                facecolor='green',
                edgecolor='white',
                lw=2,
                zorder=10):
    x_offset = (sub_state[0] + .5) / sub_cols
    y_offset = (sub_state[1] + .5) / sub_rows

    circ = plt.Circle((state[0] + x_offset, state[1] + y_offset),
                      r, facecolor=facecolor, edgecolor=edgecolor,
                      lw=lw, zorder=zorder)
    ax.add_artist(circ)
    return circ


def plot_rectangle(state, ax=None, width=.8, height=.8, orientation=None,
                   angle=0, fill=True,
                   facecolor='yellow', edgecolor='black', lw=2,
                   zorder=10):
    rect = plt.Rectangle((state[0] + (1 - width) / 2,
                          state[1] + (1 - height) / 2),
                         width, height, angle, fill=fill,
                         facecolor=facecolor, edgecolor=edgecolor,
                         lw=lw, zorder=zorder)
    ax.add_artist(rect)
    return rect


def plot_marker(state, ax, mark='x', sub_state=(1, 1),
                sub_rows=3, sub_cols=3, color='k',
                outline_color=None, outline_linewidth=2):
    x_offset = (sub_state[0] + .5) / sub_cols
    y_offset = (sub_state[1] + .5) / sub_rows

    mymarker = ax.text(state[0] + x_offset, state[1] + y_offset,
                       mark, color=color, fontsize='xx-large')

    if outline_color:
        mymarker.set_path_effects([path_effects.Stroke(
            linewidth=outline_linewidth, foreground=outline_color),
            path_effects.Normal()])
    return mymarker


def visualize_taxicab_transition(ax=None,
                                 width=6, height=6,
                                 locations=None,
                                 walls=None,
                                 taxi=None,
                                 passengers=None,
                                 action=None,
                                 passenger_colors=None,
                                 taxi_color=None):
    # draw tiles
    tiles = list(product(range(width), range(height)))
    if locations is None:
        locations = []
    ax = visualize_locs(
        ax=ax, states=tiles,
        tile_color={t: 'grey' for t in locations})

    # draw walls
    if walls is None:
        walls = []
    recode_walls = lambda w: {'north': '^', 'south': 'v',
                              'east': '>', 'west': '<'}[w]
    walls = [(s, recode_walls(w)) for s, w in walls]
    visualize_walls(ax=ax, walls=walls)

    # draw taxi
    edgecolor = 'black'
    arrow_width = .25
    arrow_len = .4
    tx, ty = taxi.location
    if action == 'dropoff':
        edgecolor = 'pink'
    elif action == 'pickup':
        edgecolor = 'lightblue'
    elif action in ['north', 'south', 'east', 'west']:
        arrowx = {'east': 1, 'west': -1}.get(action, 0)
        arrowy = {'north': 1, 'south': -1}.get(action, 0)
        ax.add_patch(plt.Arrow(tx + .5, ty + .5,
                               arrowx * arrow_len, arrowy * arrow_len,
                               width=arrow_width,
                               color='black', zorder=11))
    plot_rectangle(taxi.location, ax=ax, width=.5, height=.5,
                   edgecolor=edgecolor)

    # draw passengers and their destinations
    if passenger_colors is None:
        passenger_colors = ['g', 'b', 'r', 'purple', 'orange']

    # group by locations
    pass_radius = .12
    p_locs = {}
    for p, pcolor in zip(passengers, passenger_colors):
        if p.in_car:
            plot_circle(p.location, ax=ax, r=pass_radius,
                        sub_state=(1, 1), facecolor=pcolor,
                        edgecolor='black')
        else:
            p_locs[p.location] = p_locs.get(p.location, [])
            p_locs[p.location].append((p, pcolor))

    non_car_sstates = []
    for yx in product([2, 1, 0], [0, 1, 2]):
        xy = (yx[1], yx[0])
        if xy != (1, 1):
            non_car_sstates.append(xy)
    for loc, plist in p_locs.items():
        for sstate, (p, pcolor) in zip(non_car_sstates, plist):
            plot_circle(loc, ax=ax, r=pass_radius,
                        sub_state=sstate, facecolor=pcolor,
                        edgecolor='white')

    # draw destinations
    dests = {}
    for p, pcolor in zip(passengers, passenger_colors):
        dests[p.destination] = dests.get(p.destination, [])
        dests[p.destination].append((p, pcolor))

    dest_sstates = []
    for yx in product([0, 1, 2], [0, 1, 2]):
        xy = (yx[1], yx[0])
        if xy != (1, 1):
            dest_sstates.append(xy)
    for dest, plist in dests.items():
        for sstate, (p, pcolor) in zip(dest_sstates, plist):
            plot_marker(dest, ax=ax, mark="x",
                        sub_state=sstate,
                        color=pcolor, outline_color='white',
                        outline_linewidth=2)


def animate_transitions(taximdp, traj, filename,
                        move_interval=1000, fig=None):
    '''
    traj is a list of (state, action, next_state) tuples
    '''
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    def animate(frame):
        step, s_ns = frame
        ax.clear()
        if s_ns == 's':  # plot the state
            s_to_plot = traj[step][0]
            a = traj[step][1]
        elif s_ns == 'ns':
            s_to_plot = traj[step][2]
            a = None

        visualize_taxicab_transition(
            ax=ax,
            action=a,
            width=taximdp.width,
            height=taximdp.height,
            locations=taximdp.locs,
            walls=taximdp.walls,
            taxi=s_to_plot.taxi,
            passengers=s_to_plot.passengers)

    frames = list(product(range(len(traj)), ('s', 'ns')))

    ani = animation.FuncAnimation(fig=fig,
                                  func=animate,
                                  frames=frames,
                                  interval=move_interval)
    Writer = animation.writers['ffmpeg']
    writer = Writer(metadata=dict(artist='Mark Ho'),
                    bitrate=1800)
    ani.save(filename, writer=writer)
    return ani