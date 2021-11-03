from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess
from msdm.core.problemclasses.mdp.tabularmdp import TabularMarkovDecisionProcess
from msdm.core.distributions import DictDistribution

class CanonicalMDP(MarkovDecisionProcess):
    '''
    CanonicalMDP is a wrapper class of MDP instances that appropriately implements
    is_terminal(state) and actions(state) in terms of the simplest primitives
    typically used to describe MDPs (i.e. just the reward and transition functions).

    is_terminal(state) is a predicate used to denote absorbing states. Following
    conventions in the literature, an absorbing state transitions to itself regardless
    of action taken, and the reward received when doing this is always 0.

    actions(state) is a per-state list of actions used to denote actions that
    are valid to take at each state. When an action is invalid at a state (i.e. it is
    not in actions(state)), then the use of the action results in a transition to the
    same state, and an infinite negative reward is incurred. Importantly, we also
    ensure that the next_state_dist(s, a) function of an MDP will never be called with
    an invalid action.
    '''
    def __init__(self, mdp: MarkovDecisionProcess):
        self.mdp = mdp

    # Just copy the original MDP here.
    @property
    def discount_rate(self):
        return self.mdp.discount_rate
    def initial_state_dist(self):
        return self.mdp.initial_state_dist()
    def is_terminal(self, s):
        return self.mdp.is_terminal(s)
    def actions(self, s):
        return self.mdp.actions(s)

    def next_state_dist(self, s, a):
        if (
            a not in self.mdp.actions(s) or
            self.mdp.is_terminal(s)
        ):
            return DictDistribution.deterministic(s)
        return self.mdp.next_state_dist(s, a)

    def reward(self, s, a, ns):
        if a not in self.mdp.actions(s):
            return -float('inf')
        if (
            # Intentionally using our definition of next_state_dist since it
            # incorporates the definitions of actions/is_terminal
            self.next_state_dist(s, a).prob(ns) == 0 or
            self.mdp.is_terminal(s)
        ):
            return 0
        return self.mdp.reward(s, a, ns)

class CanonicalTabularMDP(CanonicalMDP, TabularMarkovDecisionProcess):
    def __init__(self, mdp: TabularMarkovDecisionProcess):
        super().__init__(mdp)
