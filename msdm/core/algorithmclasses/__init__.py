from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.problemclasses.stochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import SingleAgentPolicy

class Algorithm(ABC):
    """Abstract superclass for all algorithms"""
    pass

class Result(ABC):
    """Abstract superclass for all result objects"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

class Plans(Algorithm):
    @abstractmethod
    def plan_on(self, problem: ProblemClass) -> Result:
        pass

class Learns(Algorithm):
    @abstractmethod
    def train_on(self, problem: ProblemClass) -> Result:
        pass

class TabularMultiAgentLearner(Learns):
    """
    Superclass for Tabular MultiAgent Algorithms
    """
    
    def __init__(self,learning_agents:Iterable,other_policies:dict,num_episodes=200,
                 learning_rate=.1,discount_rate=1.0,
                 epsilon=0.0,default_q_value=0.0):
        self.learning_agents = learning_agents 
        self.other_agents = list(other_policies.keys())
        self.other_policies = other_policies 
        self.num_episodes = num_episodes 
        self.lr = learning_rate 
        self.dr = discount_rate 
        self.eps = epsilon 
        self.default_q_value = default_q_value 
    
    def step(self,problem:TabularStochasticGame,state,actions):
        """
        Executes one step in the environment, returning the state, actions, rewards and next state
        
        inputs: 
        ::problem:: the environment object 
        ::state:: Current state object 
        ::actions:: Hashable[agent_name -> Hashable[{'x','y'} -> {1,0,-1}]]
        
        outputs:
        (state,actions,jr,nxt_st). state and actions are same as those passed in. 
        """
        nxt_st = problem.next_state_dist(state,actions).sample()
        jr = problem.joint_rewards(state,actions,nxt_st)
        return state,actions,jr,nxt_st
    
    def pick_action(self,curr_state,q_values,problem):
        """
        Picks actions using epsilon-greedy scheme for each agent
        
        inputs:
        ::curr_state:: current state object 
        ::q_values:: Hashable[agent_name -> Hashable[state -> Hashable[action -> float]]]
        ::problem:: environment object 
        
        outputs:
        ::actions:: Hashable[agent_name -> Hashable[{'x','y'} -> {1,0,-1}]]
        """
        actions = {agent_name: None for agent_name in self.learning_agents}
        for agent_name in self.learning_agents:
            indiv_actions = list(problem.joint_actions(curr_state)[agent_name])
            # Chooses randomly among maximum actions 
            max_val = max(q_values[agent_name][curr_state].items(),key=lambda k:k[1])[1]
            max_acts = []
            for indiv_act in q_values[agent_name][curr_state].items():
                if indiv_act[1] == max_val:
                    max_acts.append(indiv_act[0])
            max_act = np.random.choice(max_acts)
            max_act = max_act[agent_name]
            # Choose action using epsilon-greedy policy 
            action = np.random.choice([max_act,indiv_actions[np.random.choice(len(indiv_actions))]],p=[1-self.eps,self.eps])
            actions[agent_name] = action

        # Getting actions for friendly agents 
        for agent in self.other_agents:
            actions[agent] = self.other_policies[agent].action_dist(curr_state).sample()
        return actions 
    
    def compute_deterministic_policy(self,q_values,problem):
        """
        Turns the Q-values for each learning agent into a deterministic policy
        
        inputs: 
        ::q_values:: Hashable[agent_name -> Hashable[state -> Hashable[action -> float]]]
        ::problem:: Environment object 
        
        outputs: 
        ::pi:: Hashable[learning_agent -> SingleAgentPolicy]
        """
        init_action_list = problem.joint_actions(problem.initial_state_dist().sample())
        pi = AssignmentMap()
        for agent in q_values:
            pi[agent] = AssignmentMap()
            for state in q_values[agent]:
                pi[agent][state] = AssignmentMap()
                # Picks randomly among maximum actions 
                max_val = max(q_values[agent][state].items(),key=lambda k:k[1])[1]
                max_acts = []
                for indiv_act in q_values[agent][state].items():
                    if indiv_act[1] == max_val:
                        max_acts.append(indiv_act[0])
                max_act = np.random.choice(max_acts)[agent]
                for action in q_values[agent][state]:
                    if action[agent] == max_act:
                        pi[agent][state][action[agent]] = 1.0
                    else:
                        pi[agent][state][action[agent]] = 0.0 
            pi[agent] = SingleAgentPolicy(agent,problem.state_list,list(init_action_list[agent]),problem.joint_action_list,pi[agent],q_values[agent])
        return pi 

    
    @abstractmethod
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        """
        Abstract method called to determine the update value to make for the Q-values after a step 
        in the environment. Should return the amount to add to the q-value for the passed in state 
        and reward. 
        """
        pass 