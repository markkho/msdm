from collections import defaultdict, namedtuple
from msdm.domains import GridWorld
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess

Location = namedtuple("Location", "x y")
GridAction = namedtuple("GridAction", "dx dy")

class GridMDP(TabularMarkovDecisionProcess):
    """
    Generic class for MDPs that can be expressed as grids.
    This requires specifying `next_state_dist`,
    `initial_state_dist`, `is_terminal`, and `reward`
    methods.
    """
    def __init__(self, grid):
        """
        Parameters
        ----------
        grid : str
            A multi-line string representing a grid MDP.
            Leading and trailing whitespace is removed.
            Each column and row is parsed into a
            Location(x, y) and feature map.
        """
        self._grid = tuple(tuple(list(r.strip())) for r in grid.strip().split('\n')[::-1])
        self._loc_features = {}
        self._feature_locs = defaultdict(list)
        for y, row in enumerate(self._grid):
            for x, f in enumerate(row):
                self._loc_features[Location(x, y)] = f
                self._feature_locs[f].append(Location(x, y))
        self._width = len(self.grid[0])
        self._height = len(self.grid)

    def feature_at(self, xy):
        return self._loc_features[xy]

    def locations_with(self, f):
        return tuple(self._feature_locs[f])

    def actions(self, s):
        return (
            GridAction(0, -1),
            GridAction(0, 1),
            GridAction(1, 0),
            GridAction(-1, 0)
        )

    @property
    def grid(self):
        return self._grid

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
