from itertools import product

from pyrlap.hierarchicalrl.ham.ham import AbstractMachine, \
    HierarchyOfAbstractMachines
from pyrlap.domains.gridworld import GridWorld
from pyrlap.domains.taxicab.utils import get_building_walls

gw = GridWorld(
    width=7, height=7,
    walls=get_building_walls(
        corners=[(1, 1), (6, 1), (6, 6), (1, 6)],
        exits=[((1, 5), '^'), ((5, 1), '>')]
    ),
    absorbing_states=[(0, 6), ],
    reward_dict = {(0, 6): 100},
    step_cost = -1,
    init_state=(4, 1)
)

outside_states = {
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)
}

inside_states = [s for s in product(range(7), range(7)) 
                 if s not in outside_states]

class Root(AbstractMachine):
    def call(self, s, stack):
        return [
            ('go_outside', ()),
            ('goto_goal', ())
        ]

class GoOutside(AbstractMachine):
    def pseudo_reward(self, s, stack):
        if s in outside_states:
            return 10
        return 0
    
    def is_terminal(self, s, stack):
        if s in outside_states:
            return True
        return False
            
    def call(self, s, stack):
        return [('^', ()), ('v', ()), ('<', ()), ('>', ())]

class GoToGoal(AbstractMachine):
    def is_terminal(self, s, stack):
        if s == (0, 6):
            return True
        if s not in outside_states:
            return True
        return False
    
    def call(self, s, stack):
        return [('^', ()), ('v', ()), ('<', ()), ('>', ())]

simple_insideoutside = HierarchyOfAbstractMachines(
    mdp=gw,
    abstract_machines={
        'root': Root,
        'go_outside': GoOutside,
        'goto_goal': GoToGoal
    }
)