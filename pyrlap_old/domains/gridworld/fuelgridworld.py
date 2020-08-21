from collections import namedtuple
from random import uniform

from pyrlap_old.domains.gridworld import GridWorld
from pyrlap_old.core.mdp import MDP

State = namedtuple("State", "loc fuel")

class FuelGridWorld(MDP):
    move_actions = ['^', 'v', '<', '>']
    wait_actions = ['x', '%']
    def __init__(self,
                 gw: GridWorld,
                 init_fuel=1.0,
                 move_cost_range=[.1, .2],
                 wait_fuel_range=[.3, .5],
                 nofuel_penalty=-100):
        self.gw = gw
        self.move_cost_range = move_cost_range
        self.wait_fuel_range = wait_fuel_range
        self.init_fuel = init_fuel
        self.nofuel_penalty = nofuel_penalty

    def get_init_state(self):
        return State(self.gw.get_init_state(), self.init_fuel)

    def is_terminal(self, s):
        loc, fuel = s
        return self.gw.is_terminal(loc)

    def is_absorbing(self, s):
        loc, fuel = s
        if fuel < 0:
            return True
        return self.gw.is_absorbing(loc)

    def is_terminal_action(self, a):
        return self.gw.is_terminal_action(a)

    def transition(self, s, a):
        if self.is_absorbing(s):
            return State(self.gw.terminal_state, 0.0)
        loc, fuel = s
        if fuel > 0 and a in self.__class__.move_actions:
            new_fuel = fuel - uniform(*self.move_cost_range)
        elif fuel > 0 and a in self.__class__.wait_actions:
            new_fuel = fuel + uniform(*self.wait_fuel_range)
        else:
            new_fuel = fuel
        new_loc = self.gw.transition(loc, a)
        return State(new_loc, new_fuel)

    def reward(self, s=None, a=None, ns=None):
        loc, fuel = ns
        gw_r = self.gw.reward(s.loc, a, ns.loc)
        if fuel < 0:
            return  gw_r + self.nofuel_penalty
        return gw_r

    def available_actions(self, s=None):
        if s is None:
            return self.gw.available_actions()
        return self.gw.available_actions(s.loc)