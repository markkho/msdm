from itertools import product
from msdm.domains.gridworld.mdp import GridWorld
from msdm.domains.modified.stickyaction import StickyActionMDP

class StickyActionGridWorld(StickyActionMDP):
    def __init__(self, *args, switch_cost=-1, **kwargs):
        """Gridworld with a small penalty for switching actions"""
        gw = GridWorld(*args, **kwargs)
        super().__init__(gw,
                         init_action={'dx': 0, 'dy': 0},
                         switch_cost=switch_cost)
        self._actions = gw._actions
        self._states = [{'groundState': s, 'curAction': a} for s, a in product(gw._states, gw._actions)]
    
    def __getattr__(self, attr):
        return getattr(self.mdp, attr)
   
    def plot(self, *args, **kwargs):
        return self.mdp.plot(*args, **kwargs)