from msdm.core.algorithmclasses import Result
from msdm.algorithms.multiagentqlearning import TabularMultiAgentQLearner
from msdm.core.stochasticgame import TabularStochasticGame
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
                 learning_rate=.1,discount_rate=1.0,
                 epsilon=0.0,epsilon_decay=1.0,default_q_value=0.0,
                 show_progress=False,alg_name="FFQ-Learning",render=False,render_from=0): 
        super().__init__(learning_agents,other_policies,num_episodes,
                        learning_rate,discount_rate,epsilon,epsilon_decay,default_q_value,
                         all_actions=True,show_progress=show_progress,
                         alg_name=alg_name,render=render,render_from=render_from)
        self.friends = friends 
        self.foes = foes 
        self.equilibria = []
    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        if problem.is_terminal(next_state):
            return self.lr*joint_rewards[agent_name]
        # Pure friend-Q case:
        if len(self.foes[agent_name]) == 0:
            ffq_equilibrium = max(q_values[agent_name][next_state].items(),key=lambda x:x[1])[1]
        else:
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
                    payoff_matrix[i][j] = q_vals[joint_action]
            cvxopt.solvers.options['show_progress'] = False
            cvx_payoff_matrix = cvxopt.matrix(payoff_matrix.T)    
            pi = variable(len(friendly_actions),"policy")
            c1 = (cvxopt.modeling.sum(pi) == 1.0)
            c2 = (pi >= 0.0)
            c4 = (pi <= 1.0)
            minimax_value = variable(1,"minimax_value")
            c3 = (minimax_value >= cvx_payoff_matrix*pi)
            constraints = [c1,c2,c3,c4]
            lp = op(-minimax_value,constraints)
            lp.solve()
            policy = np.array(pi.value)
            expected_val = np.amin(np.dot(payoff_matrix.T,policy))
            ffq_equilibrium = expected_val
            self.equilibria.append((ffq_equilibrium,next_state))
        q_del = (joint_rewards[agent_name] + self.dr*ffq_equilibrium)
        return q_del 
    
