import pytest 

from msdm.core.semimdp.semimdp import SemiMarkovDecisionProcess
from msdm.core.mdp.policy import FunctionalPolicy
from msdm.core.mdp.mdp import MarkovDecisionProcess
from msdm.core.distributions import DictDistribution
from msdm.core.semimdp.option import Option, PlanToSubgoalOption, augment
from msdm.core.exceptions import AlgorithmException
from msdm.algorithms import ValueIteration
from msdm.tests.domains import AbsorbingStateTester, DeterministicCounter, DeterministicUnreachableCounter, GNTFig6_6, \
    GeometricCounter, VaryingActionNumber, DeadEndBandit, TiedPaths, LineWorld, \
    RussellNorvigGrid_Fig17_3, PositiveRewardCycle, RussellNorvigGrid, SlipperyMaze

class SimpleOption(Option):
    _n_instances = 0
    def __init__(self, policy : FunctionalPolicy, terminal_states : list, max_steps : int):
        self.policy = policy
        self.name = f"{self.__class__}_{self.__class__._n_instances}"
        self.__class__._n_instances += 1
        self.max_steps = max_steps
        self.terminal_states = terminal_states
    def is_terminal(self, s):
        return s in self.terminal_states
    def is_initial(self, s) -> bool:
        return True

def absorbing_state_list(mdp: MarkovDecisionProcess):
    return set([s for s in mdp.reachable_states() if mdp.is_absorbing(s)])

def reachable_non_absorbing_reward_function(mdp: MarkovDecisionProcess):
    rf = {}
    for s in mdp.reachable_states():
        if mdp.is_absorbing(s):
            continue
        for a in mdp.actions(s):
            for ns, _ in mdp.next_state_dist(s, a).items():
                rf[(s, a, ns)] = mdp.reward(s, a, ns)
    return rf

def test_MDP_augmented_with_absorbing_states_and_clipped_pseudoreward():
    mdp = LineWorld(line=".is..i....g", discount_rate=.99)
    subgoals = set([i for i, c in enumerate(mdp.line) if c == 'i'])
    max_nonterminal_pseudoreward = -2
    def clipped_pseudoreward(s, a, ns):
        real_reward = mdp.reward(s, a, ns)
        if ns in subgoals:
            return real_reward
        if real_reward > max_nonterminal_pseudoreward:
            return max_nonterminal_pseudoreward
        return real_reward
    sub_mdp = augment(
        mdp=mdp,
        is_absorbing=lambda s: s in subgoals,
        reward=clipped_pseudoreward,
    )
    assert sub_mdp.reachable_states() == {1, 2, 3, 4, 5}
    assert sub_mdp.state_list == mdp.state_list, \
        "sub_mdp should inherit state_list from mdp"
    assert subgoals == absorbing_state_list(sub_mdp), \
        "sub_mdp should have absorbing states at subgoals"
    assert absorbing_state_list(mdp) != absorbing_state_list(sub_mdp), \
        "sub_mdp should not have absorbing states at mdp absorbing states"

    mdp_rf = reachable_non_absorbing_reward_function(mdp)
    sub_mdp_rf = reachable_non_absorbing_reward_function(sub_mdp)
    for s, a, ns in sub_mdp_rf.keys():
        transition_to_nonabsorbing = not sub_mdp.is_absorbing(ns)
        reward_is_greater = mdp_rf[(s, a, ns)] > max_nonterminal_pseudoreward
        if transition_to_nonabsorbing and reward_is_greater:
            assert sub_mdp_rf[(s, a, ns)] == max_nonterminal_pseudoreward, (s, a, ns)

def test_Option_max_steps_timeout():
    mdp = LineWorld(line=".is..i....g", discount_rate=.99)
    subgoals = set([i for i, c in enumerate(mdp.line) if c == 'i'])
    go_right_policy = FunctionalPolicy(lambda s : DictDistribution({1: 1}))
    go_right = SimpleOption(
        policy=go_right_policy,
        terminal_states=subgoals,
        max_steps=2
    )
    with pytest.raises(AlgorithmException) as e:
        go_right.run_on(mdp, mdp.initial_state_dist().sample())
    assert "reached max steps" in str(e.value)

def test_running_Options_on_SemiMarkovDecisionProcess():
    mdp = LineWorld(line=".is..i....g", discount_rate=.99)
    subgoals = set([i for i, c in enumerate(mdp.line) if c == 'i'])
    go_right = SimpleOption(
        policy=FunctionalPolicy(lambda s : DictDistribution({1: 1})),
        terminal_states=subgoals,
        max_steps=100
    )
    go_left = SimpleOption(
        policy=FunctionalPolicy(lambda s : DictDistribution({-1: 1})),
        terminal_states=subgoals,
        max_steps=100
    )
    traj = go_right.run_on(mdp, mdp.initial_state_dist().sample())
    assert list(traj.state) == [2, 3, 4, 5]

    smdp = SemiMarkovDecisionProcess(
        mdp=mdp,
        options=[go_left, go_right],
        n_option_simulations=5
    )
    smdp.actions(smdp.initial_state_dist().sample())
    s = smdp.initial_state_dist().sample()
    go_left_res = smdp.next_state_transit_time_dist(s, go_left)
    assert go_left_res[(1, 1)] == 1.0
    go_right_res = smdp.next_state_transit_time_dist(s, go_right)
    assert go_right_res[(5, 3)] == 1.0
    go_left_exp_rew = smdp.expected_cumulative_reward(s, go_left)
    assert go_left_exp_rew == -1
    assert smdp.expected_cumulative_reward(s, go_right) == sum([-1*mdp.discount_rate**t for t in range(3)])


def test_PlanToSubgoalOption_policy_on_stochastic_domain():
    mdp = RussellNorvigGrid_Fig17_3(discount_rate=1.0)
    upper_left, upper_right = (0, 2), (3, 2)
    go_to_upper_left = PlanToSubgoalOption(
        mdp=mdp,
        subgoals=[upper_left],
        planner=ValueIteration(max_iterations=100),
        max_nonterminal_pseudoreward=mdp.step_reward,
        include_mdp_absorbing_states=True
    )
    go_to_upper_right = PlanToSubgoalOption(
        mdp=mdp,
        subgoals=[upper_right],
        planner=ValueIteration(max_iterations=100),
        max_nonterminal_pseudoreward=mdp.step_reward,
        include_mdp_absorbing_states=True
    )
    smdp = SemiMarkovDecisionProcess(
        mdp=mdp,
        options=[go_to_upper_left, go_to_upper_right],
        n_option_simulations=50,
        seed=1288
    )
    s0 = smdp.initial_state_dist().sample()
    s1, t1 = smdp.next_state_transit_time_dist(s0, go_to_upper_left).sample()
    s2, t2 = smdp.next_state_transit_time_dist(s1, go_to_upper_right).sample()
    assert (s1, s2) == (upper_left, upper_right)

def test_Option_termination_at_multiple_subgoals():
    mdp = RussellNorvigGrid_Fig17_3(discount_rate=1.0)
    upper_left, upper_right = (0, 2), (3, 2)
    act_randomly = SimpleOption(
        policy=FunctionalPolicy(lambda s : DictDistribution.uniform(mdp.actions(s))),
        terminal_states=[upper_left] + list(absorbing_state_list(mdp)),
        max_steps=1000
    )
    smdp = SemiMarkovDecisionProcess(
        mdp=mdp,
        options=[act_randomly],
        n_option_simulations=1000,
        seed=19488
    )
    s0 = smdp.initial_state_dist().sample()
    ns_dist : DictDistribution = smdp.next_state_dist(s0, act_randomly)
    assert set(ns_dist.support) == set(act_randomly.terminal_states)

    # check that seed works at level of SemiMDP
    ns_dist_2 : DictDistribution = smdp.next_state_dist(s0, act_randomly)
    assert ns_dist_2[(0, 2)] == ns_dist[(0, 2)]
    assert id(ns_dist_2) != id(ns_dist)

    smdp_new_seed = SemiMarkovDecisionProcess(
        mdp=mdp,
        options=[act_randomly],
        n_option_simulations=1000,
        seed=1299
    )
    ns_dist_new_seed = smdp_new_seed.next_state_dist(s0, act_randomly)
    assert ns_dist_new_seed[(0, 2)] != ns_dist[(0, 2)]