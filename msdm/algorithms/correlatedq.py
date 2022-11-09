from msdm.core.algorithmclasses import Result
from msdm.algorithms.multiagentqlearning import TabularMultiAgentQLearner
from msdm.core.stochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from typing import Iterable
import itertools 
import copy
from tqdm import tqdm
from cvxopt.modeling import op 
from cvxopt.modeling import variable
from cvxopt import solvers
import cvxopt
import numpy as np
from scipy.optimize import linprog


class CorrelatedQLearner(TabularMultiAgentQLearner):
    
    def __init__(self,learning_agents: Iterable,
                 other_policies:dict,num_episodes=200,
                 learning_rate=.1,discount_rate=1.0,
                 epsilon=0.0,epsilon_decay=1.0,
                 default_q_value=0.0,objective_func="Utilitarian",show_progress=False,alg_name="Correlated Q-Learning",
                render=False,render_from=0): 
        super().__init__(learning_agents,other_policies,num_episodes,
                        learning_rate,discount_rate,epsilon,epsilon_decay,
                        default_q_value,all_actions=True,
                        show_progress=show_progress,alg_name=alg_name,render=render,render_from=render_from)
        if objective_func == "Utilitarian":
            self.objective_func =  self.utilitarian_Q
        elif objective_func == "Egalitarian":
            self.objective_func = self.egalitarian_Q
        elif objective_func == "Republican":
            self.objective_func = self.republican_Q
        elif objective_func == "Libertarian":
            self.objective_func = self.libertarian_Q
        else:
            raise Exception("Please enter one of ['Utilitarian','Egalitarian','Republican','Libertarian'] for the objective_func parameter")
        self.equilibrium_type = objective_func
        self.curr_equilibrium = None
        self.curr_equilibrium_policy = None
        
    def utilitarian_Q(self,q_values,next_state,problem,agent_name=None):
        q_matrix = np.zeros((len(self.all_agents),len(problem.joint_action_list)))
        for agent_i,agent in enumerate(self.all_agents):
            for ai, action in enumerate(problem.joint_action_list):
                q_matrix[agent_i][ai] = q_values[agent][next_state][action]
        q_matrix = np.sum(q_matrix,axis=0)
        return q_matrix
    
    def egalitarian_Q(self,q_values,next_state,problem,agent_name=None):
        q_matrix = np.zeros((len(self.all_agents),len(problem.joint_action_list)))
        for agent_i,agent in enumerate(self.all_agents):
            for ai, action in enumerate(problem.joint_action_list):
                q_matrix[agent_i][ai] = q_values[agent][next_state][action]
        return np.amin(q_matrix,axis=0)
    
    def republican_Q(self,q_values,next_state,problem,agent_name=None):
        q_matrix = np.zeros((len(self.all_agents),len(problem.joint_action_list)))
        for agent_i,agent in enumerate(self.all_agents):
            for ai, action in enumerate(problem.joint_action_list):
                q_matrix[agent_i][ai] = q_values[agent][next_state][action]
        return np.amax(q_matrix,axis=0)
    
    def libertarian_Q(self,q_values,next_state,problem,agent_name):
        q_matrix = np.zeros((len(problem.joint_action_list)))
        for ai, action in enumerate(problem.joint_action_list):
            q_matrix[ai] = q_values[agent_name][next_state][action]
        return q_matrix
        
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        if problem.is_terminal(next_state):
            return self.lr*(joint_rewards[agent_name])
        if problem.is_terminal(curr_state):
            return 0.0
        
        if self.equilibrium_type != "Libertarian":
            # compute equilibrium for first agent, then reuse for others
            if agent_name == self.learning_agents[0]:
                self.curr_equilibrium_policy,self.curr_equilibrium = self.cvxopt_equilibrium(q_values,next_state,problem)
            expected_val = 0.0
            for ai, action in enumerate(problem.joint_action_list):
                q_val = q_values[agent_name][next_state][action]
                expected_val += self.curr_equilibrium_policy[ai]*q_val
            q_del = joint_rewards[agent_name] + self.dr*expected_val
            return q_del
        # Have to recompute each equilibria for lCEQ
        else:
            self.curr_equilibrium_policy,self.curr_equilibrium = self.compute_equilibrium(q_values,next_state,problem,agent_name=agent_name)
            expected_val = 0.0
            for ai, action in enumerate(problem.joint_action_list):
                q_val = q_values[agent_name][next_state][action]
                expected_val += self.curr_equilibrium_policy[ai]*q_val
            q_del = joint_rewards[agent_name] + self.dr*expected_val
            return q_del 
    
    def compute_equilibrium(self,q_values,next_state,problem,agent_name=None):
        next_actions = problem.joint_actions(next_state)
        joint_actions = problem.joint_action_list
        # Assumes all agents have same action space 
        individual_actions = list(next_actions[self.all_agents[0]])
        num_variables = len(joint_actions)
        num_inequality_constraints = len(individual_actions)*len(self.all_agents)
        num_equality_constraints = 1 
        A_ineq = np.zeros((num_inequality_constraints,len(joint_actions)))
        for ai,action in enumerate(joint_actions): 
            curr_index = 0
            for agent_i,agent in enumerate(self.all_agents):
                altered_action = copy.deepcopy(action)
                q_val = q_values[agent][next_state][action]
                for iai,indiv_action in enumerate(individual_actions):
                    if indiv_action == action[agent]:
                        curr_index +=1 
                        continue 
                    else:
                        altered_action[agent] = indiv_action
                        altered_q_val = q_values[agent][next_state][altered_action]
                        constraint_val = (q_val - altered_q_val)
                        A_ineq[curr_index,ai] = constraint_val
                        curr_index += 1
                        
        A_eq = np.ones((1,num_variables))
        b_ineq = np.zeros((num_inequality_constraints))
        c = self.objective_func(q_values,next_state,problem,agent_name)
        lp = linprog(-c,A_ub=-A_ineq,b_ub=b_ineq,A_eq=A_eq,b_eq=1)
        equilibrium = -1*lp.fun
        policy = lp.x
        return policy,equilibrium
    
    def cvxopt_equilibrium(self,q_values,next_state,problem,agent_name=None):
        cvxopt.solvers.options['show_progress'] = False
        next_actions = problem.joint_actions(next_state)
        joint_actions = problem.joint_action_list
        policies = variable(len(joint_actions))
        constraints = []
        total_sum = {agent:0.0 for agent in joint_actions[0]}
        for ai,action in enumerate(joint_actions): 
            separate_actions = problem.joint_actions(next_state)
            policy_var = policies[ai]
            for agent in action:
                indiv_actions = list(separate_actions[agent])
                altered_action = copy.deepcopy(action)
                q_val = q_values[agent][next_state][action]
                for indiv_action in indiv_actions:
                    altered_action[agent] = indiv_action
                    altered_q_val = q_values[agent][next_state][altered_action]
                    constraint_val = (q_val - altered_q_val)*policy_var
                    total_sum[agent] += constraint_val
        for agent in total_sum:
            constraints.append((total_sum[agent] >= 0.0))
        sum_constraint = (cvxopt.modeling.sum(policies) == 1.0)
        non_negative = (policies >= 0.0)
        less_than_one = (policies <= 1.0)
        constraints.append(sum_constraint)
        constraints.append(non_negative)
        constraints.append(less_than_one)
        objective_value = cvxopt.matrix(self.objective_func(q_values,next_state,problem,agent_name))
        objective_value = cvxopt.modeling.dot(objective_value,policies)
        lp = op(-objective_value,constraints)
        lp.solve()
        equilibrium = float(lp.objective.value()[0])
        policies = list(policies.value)
        return policies,equilibrium