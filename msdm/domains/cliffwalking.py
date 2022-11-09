from msdm.domains.gridmdp import GridMDP, Location
from msdm.core.distributions import DictDistribution

class CliffWalking(GridMDP):
    def __init__(self):
        grid = """
        ............
        ............
        ............
        sxxxxxxxxxxg
        """
        super().__init__(grid)
        self.discount_rate = 1.0

    def initial_state_dist(self):
        return DictDistribution.uniform(self.locations_with('s'))

    def is_absorbing(self, s):
        return self.feature_at(s) == 'g'

    def _apply_action(self, s, a):
        ns = Location(
            max(min(s.x+a.dx, self.width - 1), 0),
            max(min(s.y+a.dy, self.height - 1), 0),
        )
        return ns

    def next_state_dist(self, s, a):
        ns = self._apply_action(s, a)
        if self.feature_at(ns) == 'x':
            return DictDistribution.uniform(self.locations_with('s'))
        return DictDistribution.deterministic(ns)

    def reward(self, s, a, ns):
        ns_ = self._apply_action(s, a)
        if self.feature_at(ns_) == 'x':
            return -100
        return -1
