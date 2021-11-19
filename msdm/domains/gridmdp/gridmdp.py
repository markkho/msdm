from collections import defaultdict, namedtuple
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.utils.funcutils import cached_property

Location = namedtuple("Location", "x y")
GridAction = namedtuple("GridAction", "dx dy")

class GridMDP(TabularMarkovDecisionProcess):
    def __init__(self, grid):
        """
        Generic class for MDPs that can be expressed as grids.
        This requires specifying `next_state_dist`,
        `initial_state_dist`, `is_terminal`, and `reward`
        methods.

        Parameters
        ----------
        grid : str
            A multi-line string representing a grid MDP.
            Leading and trailing whitespace is removed.
            Each column and row is parsed into a
            Location(x, y) and feature map.
        """
        self._grid = tuple(tuple(list(r.strip())) for r in grid.strip().split('\n')[::-1])
        assert len(set([len(r) for r in self._grid])) == 1
        self._loc_features = {}
        self._feature_locs = defaultdict(list)
        self._locations = []
        for y, row in enumerate(self._grid):
            for x, f in enumerate(row):
                loc = Location(x, y)
                self._loc_features[loc] = f
                self._feature_locs[f].append(loc)
                self._locations.append(loc)
        self._width = len(self.grid[0])
        self._height = len(self.grid)
        self._feature_locs = \
            {f: tuple(locs) for f, locs in self._feature_locs.items()}

    def plot(self, feature_colors, feature_markers, ax=None):
        #avoid circular dependency
        import matplotlib.pyplot as plt
        from msdm.domains.gridmdp.plotting import GridMDPPlotter
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plotter = GridMDPPlotter(self, ax=ax)
        plotter.fill_features(
            feature_colors=feature_colors,
            default_color='w'
        )
        plotter.mark_features(feature_markers)
        plotter.plot_outer_box()
        return plotter

    def feature_at(self, xy):
        return self._loc_features.get(xy, None)

    def locations_with(self, f):
        return self._feature_locs.get(f, ())

    def actions(self, s):
        return (
            GridAction(0, -1),
            GridAction(0, 1),
            GridAction(1, 0),
            GridAction(-1, 0)
        )

    @cached_property
    def feature_list(self):
        return tuple(sorted(self._feature_locs.keys()))

    @cached_property
    def location_list(self):
        return tuple(sorted(self._loc_features.keys()))

    @property
    def grid(self):
        return self._grid

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
