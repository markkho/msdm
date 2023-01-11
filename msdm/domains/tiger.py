from msdm.core.pomdp import TabularPOMDP
from msdm.core.distributions import DictDistribution
class Tiger(TabularPOMDP):
    def __init__(self, coherence, discount_rate):
        self.coherence = coherence
        self.discount_rate = discount_rate

    def next_state_dist(self, s, a):
        if a == 'listen':
            return DictDistribution.deterministic(s)
        return self.initial_state_dist()

    def reward(self, s, a, ns):
        if a == 'listen':
            return -1
        elif s == a:
            return -100
        else:
            return 10

    def actions(self, s):
        return ['left', 'right', 'listen']

    def initial_state_dist(self):
        return DictDistribution.uniform(['left', 'right'])

    def is_absorbing(self, s):
        return False

    def observation_dist(self, a, ns):
        if a != 'listen':
            return DictDistribution({'left':.5, 'right': .5})
        elif ns == 'left':
            pleft = self.coherence
        else:
            pleft = 1 - self.coherence
        return DictDistribution(left=pleft, right=1-pleft)
