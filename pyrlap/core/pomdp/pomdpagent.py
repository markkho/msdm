from collections import namedtuple
from random import randint
from .pomdp import PartiallyObservableMarkovDecisionProcess

SANSORTuple = namedtuple("SANSORTuple", "s a ns o r")

class POMDPAgent:
    def __init__(self, pomdp : PartiallyObservableMarkovDecisionProcess):
        self.pomdp = pomdp

    def act(self, s, **kwargs):
        raise NotImplementedError

    def run(self,
            init_state=None,
            max_steps=100):
        traj = []
        if init_state is None:
            init_state = self.pomdp.get_init_state()
        s = init_state
        i = 0
        while i < max_steps:
            a = self.act(s)
            ns = self.pomdp.transition(s, a)
            o = self.pomdp.observation(a, ns)
            r = self.pomdp.reward(s, a, ns)
            traj.append(SANSORTuple(s, a, ns, o, r))
            s = ns
            if self.pomdp.is_terminal(s):
                break
            i += 1
        return traj


class RandomPOMDPAgent(POMDPAgent):
    def __init__(self, pomdp: PartiallyObservableMarkovDecisionProcess):
        super().__init__(pomdp)

    def act(self, s, **kwargs):
        aa = self.pomdp.available_actions()
        return aa[randint(0, len(aa)-1)]