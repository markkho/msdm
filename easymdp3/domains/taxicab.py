from collections import namedtuple

import numpy as np


from easymdp3.core.mdp import MDP
from easymdp3.core.util import sample_prob_dict


class TaxiCabMDP(MDP):
    # objects for hashable state
    PassengerTuple = namedtuple(
        'Passenger',
        ['location', 'destination', 'generosity', 'in_car', 'i'])
    TaxiTuple = namedtuple(
        'Taxi', ['location', 'passenger_i', 'gas'])
    StateTuple = namedtuple(
        'State', ['passengers', 'taxi'])

    # state objects
    class Passenger(object):
        n = 0

        def __init__(self, location, destination,
                     generosity=100, in_car=False,
                     i=None):
            self.location = location
            self.destination = destination
            self.generosity = generosity
            self.in_car = in_car

            if i is None:
                i = self.__class__.n
                self.__class__.n += 1
            self.i = i

        def at_destination(self):
            return self.location == self.destination

        def as_tuple(self):
            return TaxiCabMDP.PassengerTuple(
                self.location, self.destination,
                self.generosity, self.in_car, self.i)

    class Taxi(object):
        def __init__(self, location, passenger_i=-1, gas=100):
            self.location = location
            self.gas = gas
            self.passenger_i = passenger_i

        def as_tuple(self):
            return TaxiCabMDP.TaxiTuple(
                self.location, self.passenger_i, self.gas)

        def north(self):
            self.location = (self.location[0], self.location[1] + 1)

        def south(self):
            self.location = (self.location[0], self.location[1] - 1)

        def east(self):
            self.location = (self.location[0] + 1, self.location[1])

        def west(self):
            self.location = (self.location[0] - 1, self.location[1])

        def pickup(self, passenger_i):
            self.passenger_i = passenger_i

        def dropoff(self):
            if self.passenger_i >= 0:
                self.passenger_i = -1

    class State(object):
        def __init__(self,
                     passengers=None,
                     taxi=None,
                     state_tuple=None):
            if state_tuple is not None:
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
                 step_cost=-1
                 ):
        self.width = width
        self.height = height
        if walls is None:
            walls = [((0, 0), 'east'), ((1, 0), 'west'),
                      ((0, 1), 'east'), ((1, 1), 'west'),
                      ((0, 2), 'east'), ((1, 2), 'west'),
                      ((4, 0), 'east'), ((5, 0), 'west'),
                      ((4, 1), 'east'), ((5, 1), 'west'),
                      ((4, 2), 'east'), ((5, 2), 'west'),
                      ((2, 3), 'east'), ((3, 3), 'west'),
                      ((2, 4), 'east'), ((3, 4), 'west'),
                      ((2, 5), 'east'), ((3, 5), 'west')]
        self.walls = walls
        if locations is None:
            locations = [(0, 0), (2, 5), (4, 0)]
        self.locs = locations
        self.init_location = init_location

        self.step_cost = step_cost

    def get_init_state(self):
        passengers = []
        for i, loc in enumerate(self.locs):
            dest = loc
            while dest == loc:
                dest = self.locs[np.random.randint(len(self.locs))]
            p = TaxiCabMDP.Passenger(loc, dest, i=i)
            passengers.append(p)

        taxi = TaxiCabMDP.Taxi(self.init_location, gas=100)

        self.last_state = TaxiCabMDP.State(passengers, taxi)

        return self.last_state.as_tuple()

    def available_actions(self, s=None):
        return ['north', 'south', 'east', 'west', 'noop',
                'pickup', 'dropoff']

    def transition_reward_dist(self, s=None, a=None):
        if s is None:
            s = self.last_state
        else:
            s = TaxiCabMDP.State(state_tuple=s)

        r = 0

        # taxi-transitions
        if a in ['north', 'south', 'east', 'west', 'noop'] and \
                (s.taxi.location, a) not in self.walls:
            max_x = self.width - 1
            max_y = self.height - 1
            if a == 'north' and s.taxi.location[1] < max_y:
                s.taxi.north()
            elif a == 'south' and s.taxi.location[1] > 0:
                s.taxi.south()
            elif a == 'east' and s.taxi.location[0] < max_x:
                s.taxi.east()
            elif a == 'west' and s.taxi.location[0] > 0:
                s.taxi.west()
            r += self.step_cost

        elif a == 'pickup' and s.taxi.passenger_i == -1:
            p_to_pickup = None
            for i, p in enumerate(s.passengers):
                if p.location == s.taxi.location:
                    p.in_car = True
                    s.taxi.pickup(i)
                    break

        elif a == 'dropoff' and s.taxi.passenger_i >= 0:
            taxi_passenger = s.passengers[s.taxi.passenger_i]
            taxi_passenger.in_car = False
            if taxi_passenger.at_destination():
                r += taxi_passenger.generosity
            s.taxi.dropoff()

        # passenger transitions
        for p in s.passengers:
            if p.in_car:
                p.location = s.taxi.location

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