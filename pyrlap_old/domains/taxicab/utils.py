
def get_extended_walls(start, end, width=100000, height=1000000):
    """
    Returns 'pieces' of two-sided wall for rendering. Start and end are
    x, y coordinates for the *lower left* of a tile.

    :param start:
    :param end:
    :return:
    """
    sx, sy = start
    ex, ey = end

    if sx == ex:
        walls = []
        for y in range(min(sy, ey), max(sy, ey)):
            if sx > 0:
                walls.append(((sx-1, y), '>'))
            if sx < width:
                walls.append(((sx, y), '<'))
    elif sy == ey:
        walls = []
        for x in range(min(sx, ex), max(sx, ex)):
            if sy > 0:
                walls.append(((x, sy-1), '^'))
            if sy < height:
                walls.append(((x, sy), 'v'))
    else:
        raise ValueError("Wall can only extend along single axis")
    return walls


def get_building_walls(corners, exits=None,
                       entrances=None,
                       exits_are_entrances=True,
                       width=100000,
                       height=100000
                       ):
    if exits is None:
        exits = []

    walls = []
    for ci in range(len(corners)):
        start = corners[ci]
        end = corners[(ci + 1) % len(corners)]
        side = get_extended_walls(start, end)
        walls.extend(side)

    walls = list(set(walls))

    if entrances is None:
        entrances = []
    if exits_are_entrances:
        for ex in exits:
            if ex[1] == '<' and ex[0][0] > 0:
                en = ((ex[0][0] - 1, ex[0][1]), '>')
            elif ex[1] == '>' and ex[0][0] < width:
                en = ((ex[0][0] + 1, ex[0][1]), '<')
            elif ex[1] == 'v' and ex[0][1] > 0:
                en = ((ex[0][0], ex[0][1] - 1), '^')
            elif ex[1] == '^' and ex[0][1] < height:
                en = ((ex[0][0], ex[0][1] + 1), 'v')
            entrances.append(en)

    openings = []
    openings.extend(exits)
    openings.extend(entrances)
    for op in openings:
        try:
            oi = walls.index(op)
            del walls[oi]
        except ValueError:
            raise ValueError("Wall %s is not in list" % str(op))

    return walls


def get_interior_tiles(corners):
    wall_segments = get_building_walls(corners=corners,
                                       exits=[])

    minx = min([c[0] for c in corners])
    miny = min([c[1] for c in corners])
    maxx = max([c[0] for c in corners])
    maxy = max([c[1] for c in corners])
    last_col_start_interior = False

    interior_tiles = []
    for x in range(minx, maxx):
        in_building = last_col_start_interior
        if ((x - 1, miny), '>') in wall_segments:
            in_building = not in_building
        last_col_start_interior = in_building

        for y in range(miny, maxy):
            if in_building:
                interior_tiles.append((x, y))
            if ((x, y), '^') in wall_segments and y < (maxy - 1):
                in_building = not in_building

    return interior_tiles