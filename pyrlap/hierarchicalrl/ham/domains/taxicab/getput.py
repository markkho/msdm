from pyrlap.hierarchicalrl.ham.ham import AbstractMachine, \
    HierarchyOfAbstractMachines
from pyrlap.domains.taxicab import TaxiCabMDP


class Root(AbstractMachine):
    def call(self, s, stack):
        return [
            ('get', (('passenger_i', 0),)),
            ('get', (('passenger_i', 1),)),
            ('get', (('passenger_i', 2),)),
            ('put', ())
        ]


class Get(AbstractMachine):
    def is_terminal(self, s, stack, passenger_i):
        if passenger_i in s.taxi.passengers :
            return True
        return False

    def call(self, s, stack, passenger_i):
        passenger = s.passengers[passenger_i]
        return [
            ('navigate', (('dest', passenger.location),)),
            ('pickup', ())
        ]

class Put(AbstractMachine):
    def is_terminal(self, s, stack):
        if len(s.taxi.passengers) == 0:
            return True
        return False

    def call(self, s, stack):
        passenger_i = s.taxi.passengers[0]
        p = s.passengers[passenger_i]
        return [
            ('navigate', (('dest', p.destination),)),
            ('dropoff', ())
        ]

class Navigate(AbstractMachine):
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

taxicab = TaxiCabMDP()

getput_hierarchy = HierarchyOfAbstractMachines(
    mdp=taxicab,
    abstract_machines={
        'root': Root,
        'get': Get,
        'put': Put,
        'navigate': Navigate
    }
)