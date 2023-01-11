"""QMDP

"""

from msdm.core.pomdp import TabularPOMDP
from msdm.core.pomdp.policy import ValueBasedTabularPOMDPPolicy
from msdm.core.pomdp.tabularpomdp import Belief
from msdm.core.pomdp.pomdp import Action
from msdm.core.algorithmclasses import Plans, Result
from msdm.algorithms.policyiteration import PolicyIteration

class QMDPPolicy(ValueBasedTabularPOMDPPolicy):
    def __init__(self, pomdp, stateaction_values):
        super().__init__(pomdp)
        self.sa_values = stateaction_values

    def value(self, b: Belief):
        return max([self.action_value(b, a) for a in self.pomdp.action_list])

    def action_value(self, b : Belief, a : Action):
        ss, probs = b
        aval = 0
        for s, prob in zip(ss, probs):
            aval += self.sa_values[s][a]*prob
        return aval

class QMDP(Plans):
    def __init__(
        self,
        mdp_solver : Plans = None
    ):
        """
        QMDP is a heuristic approach to solving POMDPs based on
        solving the underlying MDP and then taking actions
        according to the expected value under the current belief
        (Littman, Cassandra & Kaelbling, 1995).
        It will not take "information gathering"
        actions to solve a problem.

        Parameters
        ----------
        mdp_solver : Plans
            An instance of an MDP planning algorithm that will
            be used to solve the underlying MDP.

        Returns
        -------
        Result
            A result object with a policy

        Michael Littman, Anthony Cassandra, and Leslie Kaelbling.
        "Learning policies for partially observable environments: Scaling up."
        In Proceedings of the Twelfth International Conference on
        Machine Learning, pages 362--370, San Francisco, CA, 1995.
        """
        if mdp_solver is None:
            mdp_solver = PolicyIteration()
        self.mdp_solver = mdp_solver

    def plan_on(self, pomdp : TabularPOMDP):
        # mdp solver ignores the observation function
        # TODO: somehow assert that its an mdp solver?
        mdp_res = self.mdp_solver.plan_on(pomdp)
        sa_values = mdp_res.action_value
        return Result(
            policy=QMDPPolicy(pomdp, sa_values),
            mdp_res=mdp_res
        )

if __name__ == "__main__":
    from msdm.domains.heavenorhell import HeavenOrHell
    hh = HeavenOrHell(
        coherence=.9,
        grid=
            """
            hcg
            #.#
            #s#
            """,
        discount_rate=.99,
        heaven_reward=50,
        hell_reward=-50,
    )
    res = QMDP().plan_on(hh)
    print("Action values in MDP")
    for s, av in res.mdp_res.actionvaluefunc.items():
        print(s)
        print('\t', [((a.dx, a.dy), round(v, 2)) for a, v in av.items()])

    print()
    print("Example Trajectory")
    for st in res.policy.run_on(hh):
        heaven_is_h_pre = round(sum([p for s, p in zip(*st.agentstate) if s.heaven == 'h']), 2)
        if st.nextagentstate:
            heaven_is_h_post = round(sum([p for s, p in zip(*st.nextagentstate) if s.heaven == 'h']), 2)
        else:
            heaven_is_h_post = None
        if st.state:
            avals = [((a.dx, a.dy), round(res.policy.action_value(st.agentstate, a), 2)) for a in hh.action_list]
        else:
            avals = None
        print(
            ('s', (st.state.x, st.state.y, st.state.heaven)),
            '\n\t',
            ('avals', avals),
            '\n\t',
            ('heaven_is_h_pre', heaven_is_h_pre),
            '\n\t',
            ('o', st.observation.heaven if st.observation else None),
            '\n\t',
            ('heaven_is_h_post', heaven_is_h_post),
            '\n\t',
            ('read', st.action.read if st.action else None)
        )
        print()
