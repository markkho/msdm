from easymdp3.core.hierarchicalrl import AbstractMachine, \
    HierarchyOfAbstractMachines
from easymdp3.domains.taxicab import TaxiCabMDP


class Root(AbstractMachine):
    # def state_abstraction(self, s, stack, *args, **kwargs):
    #     ps = s.passengers
    #     passenger_status = tuple([p.location == p.destination for p in ps])
    #     has_passenger = s.taxi.passenger_i >= 0
    #     return ('root',
    #             ('passenger_status', passenger_status),
    #             ('has_passenger', has_passenger))


    def call(self, s, stack):
        return [
            ('get', (('passenger_i', 0),)),
            ('get', (('passenger_i', 1),)),
            ('get', (('passenger_i', 2),)),
            ('put', ())
        ]


class Get(AbstractMachine):
    # def state_abstraction(self, s, stack, passenger_i, *args, **kwargs):
    #     target_p = s.passengers[passenger_i]
    #     at_passenger = target_p.location == s.taxi.location
    #     has_passenger = s.taxi.passenger_i >= 0
    #     return ('get',
    #             ('passenger_i', passenger_i),
    #             ('at_passenger', at_passenger),
    #             ('has_passenger', has_passenger))

    def is_terminal(self, s, stack, passenger_i, *args, **kwargs):
        if s.taxi.passenger_i == passenger_i:
            return True
        return False

    def call(self, s, stack, passenger_i):
        passenger = s.passengers[passenger_i]
        return [
            ('navigate', (('dest', passenger.location),)),
            ('pickup', ())
        ]

class Put(AbstractMachine):
    # def state_abstraction(self, s, stack, *args, **kwargs):
    #     passenger = s.passengers[s.taxi.passenger_i]
    #     p_at_dest = passenger.destination == passenger.location
    #     return ('put',
    #             ('p_at_dest', p_at_dest))

    def is_terminal(self, s, stack, *args, **kwargs):
        if s.taxi.passenger_i == -1:
            return True
        return False

    def call(self, s, stack):
        passenger_i = s.taxi.passenger_i
        p = s.passengers[passenger_i]
        return [
            ('navigate', (('dest', p.destination),)),
            ('dropoff', ())
        ]

class Navigate(AbstractMachine):
    # def state_abstraction(self, s, stack, dest, *args, **kwargs):
    #     return ('nav',
    #             ('dest', dest),
    #             ('s.taxi.location',s.taxi.location))

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