from msdm.core.algorithmclasses import Result
from msdm.algorithms.multiagentqlearning import TabularMultiAgentQLearner
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

class FriendFoeQ(TabularMultiAgentQLearner):
    
    def __init__(self,learning_agents: Iterable,
                 friends:dict,foes:dict,
                 other_policies:dict,num_episodes=200,
                 learning_rate=.9,discount_rate=1.0,
                 epsilon=0.0,default_q_value=0.0,show_progress=False,alg_name="FFQ-Learning"): 
        super().__init__(learning_agents,other_policies,num_episodes,
                        learning_rate,discount_rate,epsilon,default_q_value,
                         all_actions=True,show_progress=show_progress,alg_name=alg_name)
        self.friends = friends 
        self.foes = foes 
    
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
        c1 = (cvxopt.modeling.sum(pi) == 1.0)
        c2 = (pi >= 0.0)
        c4 = (pi <= 1.0)
        minimax_value = variable(1,"minimax_value")
        c3 = (minimax_value >= payoff_matrix*pi)
        constraints = [c1,c2,c3,c4]
        lp = op(minimax_value,constraints)
        lp.solve()
        ffq_equilibrium = float(lp.objective.value()[0])
        q_del = self.lr*(joint_rewards[agent_name] + self.dr*ffq_equilibrium)
        return q_del 
    
