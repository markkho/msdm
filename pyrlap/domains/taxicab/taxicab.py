from collections import namedtuple

import numpy as np

from pyrlap.core.mdp import MDP
from pyrlap.core.util import sample_prob_dict
from pyrlap.domains.gridworld import GridWorld
from .vis import visualize_taxicab_transition

default_walls = [((0, 0), '>'), ((1, 0), '<'),
                  ((0, 1), '>'), ((1, 1), '<'),
                  ((0, 2), '>'), ((1, 2), '<'),
                  ((4, 0), '>'), ((5, 0), '<'),
                  ((4, 1), '>'), ((5, 1), '<'),
                  ((4, 2), '>'), ((5, 2), '<'),
                  ((2, 3), '>'), ((3, 3), '<'),
                  ((2, 4), '>'), ((3, 4), '<'),
                  ((2, 5), '>'), ((3, 5), '<')]
class TaxiCabMDP(MDP):
    # objects for hashable state
    PassengerTuple = namedtuple(
        'Passenger',
        ['location', 'destination', 'generosity', 'in_car',
         'dest_switch_prob', 'i'])
    TaxiTuple = namedtuple(
        'Taxi', ['location', 'passengers', 'gas', 'max_passengers'])
    StateTuple = namedtuple(
        'State', ['passengers', 'taxi'])

    # state objects
    class Passenger(object):
        n = 0

        def __init__(self, location, destination,
                     generosity=100, in_car=False,
                     dest_switch_prob=0,
                     i=None):
            self.location = location
            self.destination = destination
            self.generosity = generosity
            self.in_car = in_car
            self.dest_switch_prob = dest_switch_prob

            if i is None:
                i = self.__class__.n
                self.__class__.n += 1
            self.i = i

        def at_destination(self):
            return not self.in_car and (self.location == self.destination)

        def as_tuple(self):
            return TaxiCabMDP.PassengerTuple(
                self.location, self.destination,
                self.generosity, self.in_car,
                self.dest_switch_prob,
                self.i)

    class Taxi(object):
        def __init__(self, location, passengers=None,
                     gas=100, max_passengers=1):
            self.max_passengers = max_passengers

            self.location = location
            self.gas = gas
            if passengers is None:
                passengers = []
            self.passengers = list(passengers)

        def as_tuple(self):
            self.passengers.sort()
            return TaxiCabMDP.TaxiTuple(
                self.location, tuple(self.passengers), self.gas,
                self.max_passengers
            )

        def north(self):
            self.location = (self.location[0], self.location[1] + 1)

        def south(self):
            self.location = (self.location[0], self.location[1] - 1)

        def east(self):
            self.location = (self.location[0] + 1, self.location[1])

        def west(self):
            self.location = (self.location[0] - 1, self.location[1])

        def pickup(self, passenger_i):
            self.passengers.append(passenger_i)

        def dropoff(self, pi=None):
            if pi is None:
                index = 0
            else:
                index = self.passengers.index(pi)
            self.passengers.pop(index)

        def has_space(self):
            return len(self.passengers) < self.max_passengers

    class State(object):
        entity_list = ['passengers', 'taxi']

        def entities_specified(self, state_tuple):
            for entity in TaxiCabMDP.State.entity_list:
                if not hasattr(state_tuple, entity):
                    return False
            return True

        def __init__(self,
                     passengers=None,
                     taxi=None,
                     state_tuple=None):
            if state_tuple is not None and self.entities_specified(state_tuple):
                passengers = []
                for ptup in state_tuple.passengers:
                    p = TaxiCabMDP.Passenger(**ptup._asdict())
                    passengers.append(p)
                ttup = state_tuple.taxi
                taxi = TaxiCabMDP.Taxi(**ttup._asdict())
            self.passengers = passengers
            self.taxi = taxi

        def as_tuple(self):
            return TaxiCabMDP.StateTuple(
                tuple([p.as_tuple() for p in self.passengers]),
                self.taxi.as_tuple())

    # main methods
    def __init__(self,
                 width=6,
                 height=6,
                 walls: "1D walls" =None,
                 locations: "pick up and drop off locations" = None,
                 init_location=(0, 3),
                 init_passengers=None,
                 step_cost=-1,
                 unique_dropoff_pickup=False,
                 max_passengers=1
                 ):
        self.width = width
        self.height = height
        self.max_passengers = max_passengers

        if walls is None:
            walls = default_walls
        self.walls = walls
        if locations is None:
            locations = [(0, 0), (2, 5), (4, 0)]
        self.locs = locations
        self.init_location = init_location

        self.step_cost = step_cost
        if init_passengers is None:
            init_passengers = [
                {'location': self.locs[0], 'destination': self.locs[2], 'i': 0},
                {'location': self.locs[1], 'destination': self.locs[0], 'i': 1},
                {'location': self.locs[2], 'destination': self.locs[1], 'i': 2},
            ]
        self.init_passengers = init_passengers
        self.n_passengers = len(init_passengers)

        self.unique_dropoff_pickup = unique_dropoff_pickup
        if unique_dropoff_pickup:
            self.NON_TERMINAL_ACTIONS =\
                 ['^', 'v', '<', '>', 'noop'] \
               + ['pickup-%d' % pi for pi in range(self.n_passengers)] \
               + ['dropoff-%d' % pi for pi in range(self.n_passengers)]
        else:
            self.NON_TERMINAL_ACTIONS = \
                ['^', 'v', '<', '>', 'noop', 'pickup', 'dropoff']

        self.TERMINAL_ACTION = 'end'
        self.TERMINAL_STATE = TaxiCabMDP.StateTuple(None, None)

    def get_init_state(self, taxi_location=None):
        passengers = [TaxiCabMDP.Passenger(**p) for p in self.init_passengers]
        if taxi_location is None:
            taxi_location = self.init_location
        taxi = TaxiCabMDP.Taxi(taxi_location, gas=100,
                               max_passengers=self.max_passengers)
        self.last_state = TaxiCabMDP.State(passengers, taxi)
        return self.last_state.as_tuple()

    def available_actions(self, s=None):
        if s is None:
            return self.NON_TERMINAL_ACTIONS + [self.TERMINAL_ACTION, ]
        elif self.is_terminal(s) or self.is_absorbing(s):
            return [self.TERMINAL_ACTION, ]
        else:
            return self.NON_TERMINAL_ACTIONS

    def _update_state_get_reward(self, s, a):
        r = self.step_cost

        # taxi-transitions
        if a in ['^', 'v', '<', '>', 'noop'] and \
                (s.taxi.location, a) not in self.walls:
            max_x = self.width - 1
            max_y = self.height - 1
            if a == '^' and s.taxi.location[1] < max_y:
                s.taxi.north()
            elif a == 'v' and s.taxi.location[1] > 0:
                s.taxi.south()
            elif a == '>' and s.taxi.location[0] < max_x:
                s.taxi.east()
            elif a == '<' and s.taxi.location[0] > 0:
                s.taxi.west()

        elif 'pickup' in a and s.taxi.has_space():
            if self.unique_dropoff_pickup:
                pi = int(a.split('-')[1])
                p = s.passengers[pi]
                if not p.at_destination() \
                        and p.location == s.taxi.location \
                        and not p.in_car:
                    p.in_car = True
                    s.taxi.pickup(pi)
            else:
                for i, p in enumerate(s.passengers):
                    if p.location == s.taxi.location and not p.at_destination():
                        p.in_car = True
                        s.taxi.pickup(i)
                        break

        elif 'dropoff' in a and len(s.taxi.passengers) >= 0:
            if self.unique_dropoff_pickup:
                pi = int(a.split('-')[1])
                p = s.passengers[pi]
                if p.in_car:
                    p.in_car = False
                    if p.at_destination():
                        r += p.generosity
                    s.taxi.dropoff(pi)
            else:
                p = s.passengers[s.taxi.passengers[0]]
                p.in_car = False
                if p.at_destination():
                    r += p.generosity
                s.taxi.dropoff()

        # passenger transitions
        for p in s.passengers:
            if p.in_car:
                p.location = s.taxi.location

            #randomly switch destination if in car
            if p.in_car and p.dest_switch_prob > 0:
                if np.random.random() < p.dest_switch_prob:
                    new_dest = p.location
                    while new_dest == p.location:
                        new_dest = self.locs[np.random.randint(len(self.locs))]
                    p.destination = new_dest

        return s, r

    def transition_reward_dist(self, s=None, a=None):
        if a == self.TERMINAL_ACTION:
            return {(self.TERMINAL_STATE, 0): 1}

        if s is None:
            s = self.last_state
        else:
            s = TaxiCabMDP.State(state_tuple=s)

        s, r = self._update_state_get_reward(s, a)

        self.last_state = s
        return {(s.as_tuple(), r): 1}

    def transition_reward(self, s=None, a=None):
        return sample_prob_dict(self.transition_reward_dist(s, a))

    def transition(self, s=None, a=None):
        ns, _ = self.transition_reward(s, a)
        return ns

    def reward(self, s=None, a=None, ns=None):
        t_dist = self.transition_reward_dist(s, a)
        r_dist = {}
        for (ns_, r), p in t_dist.items():
            if ns_ == ns:
                r_dist[r] = r_dist.get(r, 0) + p
        return sample_prob_dict(r_dist)

    def is_absorbing(self, s=None):
        if self.is_terminal(s):
            return False

        if s is None:
            s = self.last_state
        else:
            s = TaxiCabMDP.State(state_tuple=s)

        for p in s.passengers:
            if not p.at_destination():
                return False
        return True

    def is_terminal(self, s):
        if s == self.TERMINAL_STATE:
            return True
        return False

    def is_terminal_action(self, a):
        if a == self.TERMINAL_ACTION:
            return True
        return False

    def get_gridworld(self, absorbing_states):
        return GridWorld(
            width=self.width,
            height=self.height,
            walls=self.walls
        )

    def plot(self, ax=None, figsize=(10, 10)):
        return visualize_taxicab_transition(
            ax=ax, figsize=figsize,
            width=self.width, height=self.height,
            locations=self.locs,
            walls=self.walls,
            passengers=[TaxiCabMDP.Passenger(**p) for p in self.init_passengers]
        )