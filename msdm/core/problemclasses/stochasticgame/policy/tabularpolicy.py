from typing import Mapping, Iterable
import numpy as np

from msdm.core.problemclasses.stochasticgame.policy.policy import Policy, MultiAgentPolicy
from msdm.core.problemclasses.stochasticgame import TabularStochasticGame

from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.distributions import DiscreteFactorTable, Distribution
from functools import reduce

class TabularMultiAgentPolicy(MultiAgentPolicy):
    # Switch to dictionary representation and flesh out this stuff 
    def __init__(self, problem: TabularStochasticGame, single_agent_policies: dict):
        self._states = problem.state_list
        self._joint_actions = problem.joint_action_list
        policydict = AssignmentMap()
        for si, s in enumerate(problem.state_list):
            policydict[s] = AssignmentMap()
            for agent in single_agent_policies:
                policydict[s][agent] = AssignmentMap()
                for ai, a in enumerate(self._joint_actions):
                    # Do we need the condition here? Or could the policy matrix for each agent just have the probabilities?
                    # Then it could support stochastic policies as well
                    policydict[s][agent][a] = single_agent_policies[agent].policy_matrix[si,ai]
#             for ai, a in enumerate(actions):
#                 if policymatrix[si, ai] > 0:
#                     policydict[s][a] = policymatrix[si, ai]
        self._policydict = policydict

    def evaluate_on(self, problem: TabularStochasticGame) -> Mapping:
        # do policy evaluation
        raise NotImplementedError

    def joint_action_dist(self, s) -> Distribution:
        adists = []
        for agent in self._policydict[s]:
            states,probs = zip(*self._policydict[s][agent].items())
            adist = DiscreteFactorTable(support=states,probs=probs)
            adists.append(adist)
        return reduce(lambda a, b: a & b, adists)
    
    @property
    def state_list(self):
        return self._states

    @property
    def action_list(self):
        return self._actions

    @property
    def policy_dict(self) -> Mapping:
        return self._policydict

    
class SingleAgentPolicy:
    
    def __init__(self,agent_name, policy_matrix=None):
        self._agent_name = agent_name 
        self._policy_matrix = policy_matrix
    
    @property
    def agent_name(self):
        return self._agent_name 
    
    @property
    def policy_matrix(self):
        return self._policy_matrix       
    
#     @property
#     def policy_dict(self):
#         return self._policy_dict

