from msdm.core.algorithmclasses import Result
from msdm.core.algorithmclasses import TabularMultiAgentLearner
from msdm.core.problemclasses.stochasticgame.tabularstochasticgame import TabularStochasticGame
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from msdm.core.assignment.assignmentmap import AssignmentMap
from tqdm import tqdm
from typing import Iterable
import numpy as np 
import itertools 
from scipy.special import softmax

class MultiAgentQLearning(TabularMultiAgentLearner):
    
    def __init__(self,learning_agents:Iterable,other_policies:dict,
                 num_episodes=200,learning_rate=.9,discount_rate=1.0,
                 epsilon=0.0,default_q_value=0.0):
        super().__init__(learning_agents,other_policies,num_episodes,learning_rate,
                        discount_rate,epsilon,default_q_value)
    
    def train_on(self, problem: TabularStochasticGame) -> Result:

        # initialize Q values for each agent using q learning
        res = Result()
        res.Q = {agent_name: AssignmentMap() for agent_name in self.learning_agents}
        
        for state in problem.state_list:
            for agent_name in self.learning_agents:
                res.Q[agent_name][state] = AssignmentMap()
                joint_actions = problem.joint_actions(state)
                ja_keys,ja_values = zip(*joint_actions.items())
                all_joint_actions = [dict(zip(ja_keys, list(v))) for v in itertools.product(*ja_values)]
                for joint_action in all_joint_actions:
                    res.Q[agent_name][state][joint_action] = self.default_q_value        

        for i in tqdm(range(self.num_episodes),desc="Training with Q-Learning"):
            curr_state = problem.initial_state_dist().sample()
            while not problem.is_terminal(curr_state):
                # Choose action 
                actions = self.pick_action(curr_state,res.Q,problem)
                
                #Choose action 
                curr_state,actions,jr,nxt_st = self.step(problem,curr_state,actions)
                
                # update q values for each agent 
                for agent_name in self.learning_agents:
                    q_del = self.update(agent_name,actions,res.Q,jr,curr_state,nxt_st,problem)
                    res.Q[agent_name][curr_state][actions] += q_del
                curr_state = nxt_st
                
        # Converting to dictionary representation of deterministic policy
        pi = self.compute_deterministic_policy(res.Q,problem)
        
        # add in non_learning agents 
        for agent in self.other_agents:
            pi[agent] = self.other_policies[agent]
            
        # create result object
        res.problem = problem
        res.policy = {}
        res.policy = res.pi = TabularMultiAgentPolicy(problem, pi,self.dr)
        return res
    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        q_del = joint_rewards[agent_name]     
        q_del += self.dr*(max(q_values[agent_name][next_state].items(),key=lambda k:k[1])[1])
        q_del -= q_values[agent_name][curr_state][actions]
        q_del *= self.lr
        return q_del 
    
