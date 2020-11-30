from msdm.core.algorithmclasses import TabularMultiAgentLearner, Result
from msdm.core.problemclasses.stochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from tqdm import tqdm
import itertools 
from typing import Iterable
import numpy as np 
from cvxopt.modeling import op 
from cvxopt.modeling import variable
import cvxopt

class FriendFoeQ(TabularMultiAgentLearner):
    
    def __init__(self,learning_agents: Iterable,
                 friends:dict,foes:dict,
                 other_policies:dict,num_episodes=200,
                 learning_rate=.9,discount_rate=1.0,
                 epsilon=0.0,default_q_value=0.0): 
        super().__init__(learning_agents,other_policies,num_episodes,
                        learning_rate,discount_rate,epsilon,default_q_value)
        self.friends = friends 
        self.foes = foes 
    
    def train_on(self,problem:TabularStochasticGame) -> Result:
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

        for i in tqdm(range(self.num_episodes)):
            curr_state = problem.initial_state_dist().sample()
            while not problem.is_terminal(curr_state):
                # Choose joint action 
                actions = self.pick_action(curr_state,res.Q,problem)
                # Take a step in the environment
                curr_state,actions,jr,nxt_st = self.step(problem,curr_state,actions)

                # update q values for each agent 
                for agent_name in self.learning_agents:
                    q_del = self.update(agent_name,actions,res.Q,jr,curr_state,nxt_st,problem)
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
        res.policy = res.pi = TabularMultiAgentPolicy(problem, pi,self.dr)
        return res
    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        next_actions = problem.joint_actions(next_state)
        friendly_actions = [list(next_actions[agent_name])]
        foe_actions = []
        friend_order = {agent_name:0}
        foe_order = {}
        for i,friend in enumerate(self.friends[agent_name]):
            friend_actions = list(next_actions[friend])
            friendly_actions.append(friend_actions)
            friend_order[friend] = i+1
            
        for i,foe in enumerate(self.foes[agent_name]):
            foe_action = list(next_actions[foe])
            foe_actions.append(foe_action)
            foe_order[foe] = i
            
        friendly_actions = list(itertools.product(*friendly_actions))
        foe_actions = list(itertools.product(*foe_actions))

        payoff_matrix = np.zeros((len(friendly_actions),len(foe_actions)))
        q_vals = q_values[agent_name][next_state]
        for i,action in enumerate(friendly_actions):                
            for j,foe_action in enumerate(foe_actions):
                joint_action = {}
                for friend in friend_order:
                    joint_action[friend] = action[friend_order[friend]]
                for foe in foe_order:
                    joint_action[foe] = foe_action[foe_order[foe]]
                payoff_matrix[i,j] = q_vals[joint_action]
        
        cvxopt.solvers.options['show_progress'] = False
        payoff_matrix = cvxopt.matrix(payoff_matrix.T)    
        pi = variable(len(friendly_actions),"policy")      
        c1 = (-cvxopt.modeling.sum(pi) == 1.0)
        c2 = (pi >= 0.0)
        c4 = (pi <= 1.0)
        minimax_value = variable(1,"minimax_value")
        c3 = (payoff_matrix*pi >= minimax_value)
        constraints = [c1,c2,c3,c4]
        lp = op(minimax_value,constraints)
        lp.solve()
        ffq_equilibrium = float(lp.objective.value()[0])

        q_del = self.lr*(joint_rewards[agent_name] + self.dr*ffq_equilibrium)
        return q_del 
    
