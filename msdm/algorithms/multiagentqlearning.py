from msdm.core.algorithmclasses import Result
from msdm.core.algorithmclasses import Learns
from msdm.core.problemclasses.stochasticgame.tabularstochasticgame import TabularStochasticGame
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from msdm.core.assignment.assignmentmap import AssignmentMap
from tqdm import tqdm
from typing import Iterable
import numpy as np 
import itertools 
from scipy.special import softmax

    
class TabularMultiAgentQLearner(Learns):
    
    def __init__(self,learning_agents:Iterable,other_policies:dict,num_episodes=200,
                 learning_rate=.1,discount_rate=1.0,
                 epsilon=0.0,default_q_value=0.0,all_actions=True,show_progress=False,alg_name="Q-Learning"):
        self.learning_agents = learning_agents 
        self.other_agents = list(other_policies.keys())
        self.other_policies = other_policies 
        self.num_episodes = num_episodes 
        self.lr = learning_rate 
        self.dr = discount_rate 
        self.eps = epsilon 
        self.default_q_value = default_q_value 
        self.show_progress = show_progress
        self.all_actions = all_actions 
        self.alg_name = alg_name
    
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
            for act in q_values[agent_name][curr_state].items():
                if act[1] == max_val:
                    max_acts.append(act[0])
            max_act = np.random.choice(max_acts)
            if self.all_actions:
                max_act = max_act[agent_name]
            # Choose action using epsilon-greedy policy 
            action = np.random.choice([max_act,indiv_actions[np.random.choice(len(indiv_actions))]],p=[1-self.eps,self.eps])
            actions[agent_name] = action

        # Getting actions for friendly agents 
        for agent in self.other_agents:
            actions[agent] = self.other_policies[agent].action_dist(curr_state).sample()
        return actions 
    
    def train_on(self,problem: TabularStochasticGame) -> Result:
        # initialize Q values for each agent using q learning
        res = Result()
        res.Q = {agent_name: AssignmentMap() for agent_name in self.learning_agents}
        
        for state in problem.state_list:
            for agent_name in self.learning_agents:
                if not self.all_actions:
                    res.Q[agent_name][state] = AssignmentMap()
                    indiv_actions = list(problem.joint_actions(state)[agent_name])
                    for action in indiv_actions:
                        res.Q[agent_name][state][action] = self.default_q_value
                else:
                    res.Q[agent_name][state] = AssignmentMap()
                    joint_actions = problem.joint_actions(state)
                    ja_keys,ja_values = zip(*joint_actions.items())
                    all_joint_actions = [dict(zip(ja_keys, list(v))) for v in itertools.product(*ja_values)]
                    for joint_action in all_joint_actions:
                        res.Q[agent_name][state][joint_action] = self.default_q_value        
        
        if self.show_progress:
            episodes = tqdm(range(self.num_episodes),desc="Training with " + self.alg_name)
        else:
            episodes = range(self.num_episodes)
                    
        for i in episodes:
            curr_state = problem.initial_state_dist().sample()
            while not problem.is_terminal(curr_state):
                # Choose action 
                actions = self.pick_action(curr_state,res.Q,problem)
                
                #Choose action 
                curr_state,actions,jr,nxt_st = self.step(problem,curr_state,actions)
                
                # update q values for each agent 
                for agent_name in self.learning_agents:
                    q_del = self.update(agent_name,actions,res.Q,jr,curr_state,nxt_st,problem)
                    if not self.all_actions:
                        res.Q[agent_name][curr_state][actions[agent_name]] = (1-self.lr)*res.Q[agent_name][curr_state][actions[agent_name]] + q_del
                    else:
                        res.Q[agent_name][curr_state][actions] = (1-self.lr)*res.Q[agent_name][curr_state][actions] + q_del
                curr_state = nxt_st
                
        # Converting to dictionary representation of deterministic policy
        pi = self.compute_deterministic_policy(res.Q,problem)
        
        # add in non_learning agents 
        for agent in self.other_agents:
            pi[agent] = self.other_policies[agent]
            
        # create result object
        res.problem = problem
        res.policy = {}
        res.policy = res.pi = TabularMultiAgentPolicy(problem, pi,self.dr,show_progress=self.show_progress)
        return res
    
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
                for act in q_values[agent][state].items():
                    if act[1] == max_val:
                        max_acts.append(act[0])
                max_act = np.random.choice(max_acts)
                if self.all_actions:
                    max_act = max_act[agent]
                for action in q_values[agent][state]:
                    if self.all_actions:
                        if action[agent] == max_act:
                            pi[agent][state][action[agent]] = 1.0
                        else:
                            pi[agent][state][action[agent]] = 0.0
                    else:
                        if action == max_act:
                            pi[agent][state][action] = 1.0
                        else:
                            pi[agent][state][action] = 0.0
            pi[agent] = SingleAgentPolicy(agent,problem,pi[agent],q_values[agent],self.all_actions)
        return pi 

    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        q_del = joint_rewards[agent_name]     
        q_del += self.dr*(max(q_values[agent_name][next_state].items(),key=lambda k:k[1])[1])
        if not self.all_actions:
            q_del -= q_values[agent_name][curr_state][actions[agent_name]]
        else:
            q_del -= q_values[agent_name][curr_state][actions]
        q_del *= self.lr
        return q_del 
    
