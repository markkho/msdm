from pyrlap.hierarchicalrl.ham.ham import AbstractMachine, \
    HierarchyOfAbstractMachines
from pyrlap.domains.taxicab import TaxiCabMDP

# Restricted get/put hierarchy
class Root(AbstractMachine):
    def call(self, s, stack):
        if len(s.taxi.passengers) == 0:
            for i in range(len(s.passengers)):
                if s.passengers[i].location != s.passengers[i].destination:
                    return [('get', (('passenger_i', i),)), ]
            return [('put', ()), ]
        else:
            return [('put', ()), ]

class Get(AbstractMachine):
    def is_terminal(self, s, stack, passenger_i):
        if passenger_i in s.taxi.passengers:
            return True
        return False

    def call(self, s, stack, passenger_i):
        passenger = s.passengers[passenger_i]
        if s.taxi.location == passenger.location:
            return [('pickup', ()), ]
        else:
            return [('navigate', (('dest', passenger.location),)), ]

class Put(AbstractMachine):
    def is_terminal(self, s, stack):
        if len(s.taxi.passengers) == 0:
            return True
        return False

    def call(self, s, stack):
        passenger_i = s.taxi.passengers[0]
        p = s.passengers[passenger_i]
        if s.taxi.location == p.destination:
            return [('dropoff', ()), ]
        else:
            return [('navigate', (('dest', p.destination),)), ]

class Navigate(AbstractMachine):
    def is_terminal(self, s, stack, dest, *args, **kwargs):
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
    width=6, height=6, walls=None,
    locations = [(0, 0), (2, 5)],
    init_passengers = [
        {'location': (0, 0), 'destination': (2, 5), 'i': 0},
    ]
)

restricted_getput = HierarchyOfAbstractMachines(
    mdp=taxicab,
    abstract_machines={
        'root': Root,
        'get': Get,
        'put': Put,
        'navigate': Navigate
    }
)