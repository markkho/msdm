from collections import namedtuple
from .pomdp import PartiallyObservableMarkovDecisionProcess

import numpy as np

SANSORTuple = namedtuple("SANSORTuple", "s a ns o r")
StateActionObsNodeTuple = namedtuple("StateActionObsNodeTuple",
                                     "s z a ns o nz r")

class POMDPAgent:
    def __init__(self, pomdp : PartiallyObservableMarkovDecisionProcess):
        self.pomdp = pomdp
        self._belief_state = None

    # todo: automatically track belief state

    def initialize(self, internal_state=None):
        """This can be a belief or some other internal representation."""
        # self._belief_state = self.pomdp.initial_belief()
        raise NotImplementedError

    def act(self, **kwargs):
        raise NotImplementedError

    def update(self, o: "observation"):
        # self._belief_state = self.pomdp.belief_update(self._belief_state, o)
        raise NotImplementedError

    def get_internal_state(self):
        """This can be a belief or some other internal representation."""
        raise NotImplementedError

    def run(self,
            init_state=None,
            init_internal_state=None,
            max_steps=100,
            include_internal_state=False
            ):
        traj = []
        if init_state is None:
            init_state = self.pomdp.get_init_state()
        s = init_state
        i = 0
        self.initialize(init_internal_state)
        while i < max_steps:
            a = self.act()
            ns = self.pomdp.transition(s, a)
            o = self.pomdp.observation(a, ns)
            r = self.pomdp.reward(s, a, ns)
            if include_internal_state:
                z = self.get_internal_state()
                self.update(o)
                nz = self.get_internal_state()
                traj.append(StateActionObsNodeTuple(s, z, a, ns, o, nz, r))
            else:
                self.update(o)
                traj.append(SANSORTuple(s, a, ns, o, r))
            s = ns
            if self.pomdp.is_terminal(s):
                break
            i += 1
        return traj
