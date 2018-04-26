from easymdp3.core.hierarchicalrl import AbstractMachine, \
    HierarchyOfAbstractMachines
from easymdp3.domains.taxicab import TaxiCabMDP


class Root(AbstractMachine):
    def state_abstraction(self, s, stack):
        return hash((s, tuple(stack)))

    def call(self, s, stack):
        return [
            ('get', (('passenger_i', 0),)),
            ('get', (('passenger_i', 1),)),
            ('put', ())
        ]

class Get(AbstractMachine):
    def state_abstraction(self, s, stack, passenger_i):
        return hash((s, tuple(stack)))

    def is_terminal(self, s, stack, passenger_i):
        if passenger_i in s.taxi.passengers:
            return True
        if len(s.taxi.passengers) > 0:
            return True
        return False

    def call(self, s, stack, passenger_i):
        passenger = s.passengers[passenger_i]
        return [
            ('pickup', ()),
            ('navigate', (('dest', passenger.location),))
        ]

class Put(AbstractMachine):
    def state_abstraction(self, s, stack):
        return hash((s, tuple(stack)))

    def is_terminal(self, s, stack):
        if len(s.taxi.passengers) == 0:
            return True
        return False

    def call(self, s, stack):
        passenger_i = s.taxi.passengers[0]
        p = s.passengers[passenger_i]
        return [
            ('dropoff', ()),
            ('navigate', (('dest', p.destination),))
        ]

class Navigate(AbstractMachine):
    def state_abstraction(self, s, stack, dest):
        return hash((s, tuple(stack)))

    def is_terminal(self, s, stack, dest):
        if s.taxi.location == dest:
            return True
        return False

    def call(self, s, stack, dest):
        return [
            ('v', ()),
            ('^', ()),
            ('<', ()),
            ('>', ())
        ]

taxicab = TaxiCabMDP(
    width=3, height=3, walls=[],
    locations = [(0, 0), (1, 2)],
    init_passengers = [
        {'location': (1, 2), 'destination': (0, 0), 'i': 0},
        {'location': (0, 0), 'destination': (1, 2), 'i': 1},
    ],
    init_location=(0, 2)
)

simple_getput = HierarchyOfAbstractMachines(
    mdp=taxicab,
    abstract_machines={
        'root': Root,
        'get': Get,
        'put': Put,
        'navigate': Navigate
    }
)