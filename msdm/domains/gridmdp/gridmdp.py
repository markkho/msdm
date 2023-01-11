from collections import defaultdict, namedtuple
from msdm.core.mdp import TabularMarkovDecisionProcess
from msdm.core.utils.funcutils import cached_property

Location = namedtuple("Location", "x y")
GridAction = namedtuple("GridAction", "dx dy")

class GridMDP(TabularMarkovDecisionProcess):
    def __init__(self, grid):
        """
        Generic class for MDPs that can be expressed as grids.
        This requires specifying `next_state_dist`,
        `initial_state_dist`, `is_absorbing`, and `reward`
        methods.

        Parameters
        ----------
        grid : str
            A multi-line string representing a grid MDP.
            Leading and trailing whitespace is removed.
            Each column and row is parsed into a
            Location(x, y) and feature map.
        """
        self._grid_string = grid

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
        return self.location_feature_dict.get(xy, None)

    def locations_with(self, f):
        return self.feature_locations_dict.get(f, ())

    def actions(self, s):
        return (
            GridAction(0, -1),
            GridAction(0, 1),
            GridAction(1, 0),
            GridAction(-1, 0)
        )

    @cached_property
    def grid(self):
        grid = tuple(tuple(list(r.strip())) for r in self._grid_string.strip().split('\n'))
        assert len(set([len(r) for r in grid])) == 1
        return grid

    @cached_property
    def location_feature_dict(self):
        location_features = {}
        for y, row in enumerate(self.grid[::-1]):
            for x, f in enumerate(row):
                loc = Location(x, y)
                location_features[loc] = f
        return location_features
    
    @cached_property
    def feature_locations_dict(self):
        feature_locations = defaultdict(list)
        for loc, feature in self.location_feature_dict.items():
            feature_locations[feature].append(loc)
        feature_locations = \
            {f: tuple(locs) for f, locs in feature_locations.items()}
        return feature_locations

    @cached_property
    def feature_list(self):
        return tuple(sorted(self.feature_locations_dict.keys()))

    @cached_property
    def location_list(self):
        return tuple(sorted(self.location_feature_dict.keys()))

    @property
    def width(self):
        return len(self.grid[0])

    @property
    def height(self):
        return len(self.grid)
