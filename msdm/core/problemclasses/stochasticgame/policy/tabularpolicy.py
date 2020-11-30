from typing import Mapping, Iterable
import numpy as np
from itertools import product
from copy import copy 
import sparse
from tqdm import tqdm

from msdm.core.problemclasses.stochasticgame.policy.policy import Policy, MultiAgentPolicy
from msdm.core.problemclasses.stochasticgame import TabularStochasticGame

from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.distributions import DiscreteFactorTable, Distribution
from functools import reduce

class TabularMultiAgentPolicy(MultiAgentPolicy):
    """
    Class to represent multiple agent policies combined together
    """
    
    def __init__(self, problem: TabularStochasticGame, single_agent_policies: dict,discount_rate=1.0):
        self._states = problem.state_list
        self._joint_actions = problem.joint_action_list
        self.problem = problem
        policydict = AssignmentMap()
        # Generates a policy dictionary Hashable[state -> Hashable[agent -> Hashable[actions -> probabilities]]]
        for si, s in enumerate(problem.state_list):
            policydict[s] = AssignmentMap()
            for agent in single_agent_policies:
                policydict[s][agent] = single_agent_policies[agent].policy_dict[s]
        self._policydict = policydict
        self.single_agent_policies = single_agent_policies 
        self.discount_rate = discount_rate

    def evaluate_on(self, problem: TabularStochasticGame) -> Mapping:
        # do policy evaluation
        raise NotImplementedError

    def joint_action_dist(self, s) -> Distribution:
        adists = []
        for agent in self._policydict[s]:
            actions, probs = zip(*self._policydict[s][agent].items())
            adist = DiscreteFactorTable([{agent:action} for action in actions], probs=probs)
            adists.append(adist)
        adists = reduce(lambda a, b: a & b, adists)
        return adists
    
    def projected_Q(self,agent_name,q_matrix,weight_matrix):
        """
        Computes the Q-values for each position(x,y coordinate an agent can occupy) in the problem. Averages over the q-values for all states including the agent in that position using the weight_matrix to determine weighting(usually the occupancy distribution for the poicy and enviroment).
        
        inputs: 
        ::agent_name:: string representing the name of the agent 
        ::q_matrix:: numpy matrix of size (num_states,num_actions) representing q_values for each state,action pair(not joint actions).
        ::weight_matrix:: matrix of size (num_states,num_states) used to weight the average across states
        
        outputs: 
        ::proj_q_matrix:: a matrix of size (num_positions,num_actions) representing the projected Q values for each position
        """
        indiv_actions = self.single_agent_policies[agent_name]._actions
        proj_q_matrix = np.zeros((len(self.problem.position_list),len(indiv_actions)))
        initial_state_occupancy = weight_matrix[self._states.index(self.problem.initial_state_dist().sample())]
        for si,agent_position in enumerate(tqdm(self.problem.position_list,desc="Calculating Projected Q-Values")):
            for ai, action in enumerate(indiv_actions):
                # Computes all the possible positions and action combinations for the other agents 
                positions = copy(self.problem.position_list)
                positions.remove(agent_position)
                positions = [positions for i in range(len(self.problem.agent_names)-1)]
                # works since they all have same action space at the moment 
                actions = [indiv_actions for i in range(len(self.problem.agent_names)-1)]
                position_actions = []
                for i in range(len(positions)):
                    position_actions.append(positions[i])
                    position_actions.append(actions[i])
                position_actions = product(*position_actions)
                pos_actions = [] 
                for config in position_actions:
                    # Checking if positions are the same for any pair of agents
                    for i in range(0,len(config)-1,2):
                        if i != len(config)-2:
                            if config[i] == config[i+2]:
                                break 
                        # add if no overlapping positions
                        else:
                            pos_actions.append(config)
                total_val = 0.0 
                # goes through each configuration for the current position, weighting each one using the weight matrix and 
                # adding it to the total projected value. 
                for config in pos_actions:
                    agent_names = copy(self.problem.agent_names)
                    agent_names.remove(agent_name)
                    agent_positions = {agent_name:agent_position}
                    agent_actions = {agent_name:action}
                    for i in range(0,2*(len(agent_names)),2):
                        agent = agent_names[i//2]
                        agent_positions[agent] = config[i]
                        agent_actions[agent] = config[i+1]
                    state = self.construct_state(agent_positions)
                    occupancy = initial_state_occupancy[self._states.index(state)]
                    q_val = q_matrix[self._states.index(state)][self._joint_actions.index(agent_actions)]
                    config_val = occupancy*q_val 
                    for agent in agent_names:
                        config_val *= self._policydict[state][agent][agent_actions[agent]]
                    total_val += config_val
                proj_q_matrix[si][ai] = total_val 
        return proj_q_matrix
    
    def projected_V(self,agent_name,q_matrix,weight_matrix):
        """
        Uses the projected_Q functions to compute the projected values
        """
        proj_q = self.projected_Q(agent_name,q_matrix,weight_matrix)
        proj_v = np.zeros((len(self.problem.position_list)))
        for i,position in enumerate(proj_q):
            proj_v[i] = np.max(position)
        return proj_v
            
    def construct_state(self,positions):
        """
        Converts Hashable[agent_name -> tuple(x,y)] into 
        a state object
        """
        state = {} 
        for agent in positions:
            state[agent] = {} 
            state[agent]["name"] = agent 
            state[agent]["type"] = "agent"
            state[agent]["x"] = positions[agent][0]
            state[agent]["y"] = positions[agent][1]
        return state
    
    def positionMapping(self,agent_name,q_matrix,weight_matrix):
        """
        Computes the projected Values, and then turns them into 
        a dictionary from position -> value for visualization 
        """
        projected_vals = self.projected_V(agent_name,q_matrix,weight_matrix)
        mapping = {}
        for i,position in enumerate(self.problem.position_list):
            mapping[position] = projected_vals[i]
        return mapping 
    
    def positionActionMapping(self,agent_name,q_matrix,weight_matrix):
        """
        Computes the projected Q-values, and then turns them into 
        a dictionary from position,action -> value for visualization 
        """
        projected_q_vals = self.projected_Q(agent_name,q_matrix,weight_matrix)
        mapping = AssignmentMap()
        for i,position in enumerate(self.problem.position_list):
            mapping[position] = AssignmentMap() 
            for j,action in enumerate(self.single_agent_policies[agent_name]._actions):
                mapping[position][action] = projected_q_vals[i][j]
        return mapping
        
    
    @property
    def occupancy_matrix(self):
        """
        Computes the occupancy distribution for the Markov Chain defined by the policy and environment. 
        returns a (num_state,num_state) matrix, where each row of the matrix is the expected visits 
        to each state, given that the row state is the initial one. Used for weighting in visualization. 
        """
        try:
            return self._occupancy_matrix
        except AttributeError:
            joint_policy_matrix = self.joint_policy_matrix
            transition_matrix = np.copy(self.problem.transitionmatrix)
            state_to_state_matrix = np.transpose(transition_matrix,axes=[2,0,1])
            for i in tqdm(range(len(state_to_state_matrix)),desc="Generating Occupancy Matrix"):
                sparse.elemwise(np.multiply,state_to_state_matrix[i],joint_policy_matrix)
#                 state_to_state_matrix[i] *= joint_policy_matrix
            state_to_state_matrix = np.sum(state_to_state_matrix,axis=2).T
            occupancy_matrix = np.identity(state_to_state_matrix.shape[0]) - self.discount_rate*state_to_state_matrix
            occupancy_matrix = np.linalg.inv(occupancy_matrix)
            self._occupancy_matrix = occupancy_matrix 
            return self._occupancy_matrix
        
        
    @property     
    def joint_policy_matrix(self):
        """
        Generates a matrix of size (num_states,num_joint_actions) representing the total joint policy for 
        all the agents. 
        """
        try:
            return self._joint_policy_matrix
        except AttributeError:
            self._joint_policy_matrix = np.zeros((len(self._states),len(self._joint_actions)))
            for si,state in enumerate(self._states):
                for ai,action in enumerate(self._joint_actions):
                    policy_val = 1.0 
                    for agent in action:
                        policy_val *= self._policydict[state][agent][action[agent]]
                    self._joint_policy_matrix[si][ai] = policy_val 
            return self._joint_policy_matrix
            
    
    @property
    def state_list(self):
        return self._states

    @property
    def joint_action_list(self):
        return self.joint_actions

    @property
    def policy_dict(self) -> Mapping:
        return self._policydict

    
class SingleAgentPolicy(Policy):
    
    def __init__(self,agent_name, state_list, actions,joint_actions,policy_dict,q_vals=None):
        self._agent_name = agent_name 
        self._policy_dict = policy_dict
        self._actions = actions
        self._joint_actions = joint_actions
        self._state_list = state_list
        self.q_vals = q_vals
        
    @property
    def agent_name(self):
        return self._agent_name 
        
    @property
    def policy_dict(self):
        return self._policy_dict
    
    @property 
    def q_matrix(self):
        if self.q_vals == None:
            print("Q-values not stored for this agent's policy")
            return 
        try:
            return self._q_matrix 
        except AttributeError: 
            q_mat = np.zeros((len(self._state_list),len(self._joint_actions)))
            for si,state in enumerate(self._state_list):
                for ai, action in enumerate(self._joint_actions):
                    q_mat[si][ai] = self.q_vals[state][action]
            self._q_matrix = q_mat
            return self._q_matrix
            
    @property 
    def policy_matrix(self):
        try:
            return self._policy_matrix
        except AttributeError:
            self._policy_matrix = np.zeros((len(self._state_list),len(self._actions)))
            for si,state in enumerate(self._state_list):
                adist = self.action_dist(state)
                for ai, action in enumerate(self._actions):
                    self._policy_matrix[si,ai] = adist.prob(action)
            return self._policy_matrix
    
    def action_dist(self,s) -> Distribution:
        adist = self._policy_dict[s]
        a, p = zip(*adist.items())
        return DiscreteFactorTable(support=a,probs=p)
