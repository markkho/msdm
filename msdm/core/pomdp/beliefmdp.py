from collections import defaultdict
from msdm.core.pomdp.tabularpomdp import TabularPOMDP, Belief
from msdm.core.mdp import MarkovDecisionProcess
from msdm.core.distributions import DictDistribution

class BeliefMDP(MarkovDecisionProcess):
    def __init__(self, pomdp: TabularPOMDP):
        """
        Constructs a belief MDP out of a tabular POMDP.
        See Kaelbling, Littman & Cassandra (1998) for
        details.
        """
        self.pomdp = pomdp
        self.discount_rate = pomdp.discount_rate

    def initial_state_dist(self):
        s0 = Belief(tuple(self.pomdp.state_list), tuple(self.pomdp.initial_state_vec))
        return DictDistribution.deterministic(s0)

    def is_absorbing(self, s):
        for state, prob in zip(*s):
            if (prob > 0.0) and not self.pomdp.is_absorbing(state):
                return False
        return True

    def next_state_dist(self, s, a):
        b = DictDistribution(dict(zip(*s)))
        o_dist = self.pomdp.predictive_observation_dist(b, a)
        nb_dist = defaultdict(float)
        for o, o_prob in o_dist.items():
            nb = self.pomdp.state_estimator(b, a, o)
            nb = [nb.get(e, 0.0) for e in self.pomdp.state_list]
            nb = Belief(states=tuple(self.pomdp.state_list), probs=tuple(nb))
            if o_prob > 0.0:
                nb_dist[nb] += o_prob
        return DictDistribution(nb_dist)

    def reward(self, s, a, ns):
        # note we ignore the next belief state here
        b = DictDistribution(dict(zip(*s)))
        r = 0
        for s, s_prob in b.items():
            sa_reward = 0
            for ns, ns_prob in self.pomdp.next_state_dist(s, a).items():
                sa_reward += self.pomdp.reward(s, a, ns)*ns_prob
            r += sa_reward*s_prob
        return r

    def actions(self, s):
        return tuple(self.pomdp.action_list)
