from msdm.core.algorithmclasses import Result
from msdm.algorithms.multiagentqlearning import TabularMultiAgentQLearner
from msdm.core.problemclasses.stochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from typing import Iterable
import itertools 
import copy
from tqdm import tqdm
from cvxopt.modeling import op 
from cvxopt.modeling import variable
import cvxopt


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
    
    def utilitarian_Q(self,agent_name,q_values,policy,curr_state,problem):
        agents = [] 
        agents.extend(self.learning_agents)
        agents.extend(list(self.other_policies.keys()))
        total_val = 0.0
        for agent in agents:
            for ai,action in enumerate(problem.joint_action_list):
                total_val += policy[ai]*q_values[agent][curr_state][action]
        # Negated so the LP solver will maximize instead of minimize this value
        return -total_val
    
    def egalitarian_Q(self,agent_name,q_values,policy,curr_state,problem):
        agents = [] 
        agents.extend(self.learning_agents)
        agents.extend(list(self.other_policies.keys()))
        agent_vals = []
        for agent in agents:
            agent_total = 0.0
            for ai,action in enumerate(problem.joint_action_list):
                agent_total += policy[ai]*q_values[agent][curr_state][action]
            agent_vals.append(agent_total)
        # Negated so the LP solver will maximize instead of minimize this value
        return -cvxopt.modeling.min(agent_vals)
    
    def republican_Q(self,agent_name,q_values,policy,curr_state,problem):
        agents = [] 
        agents.extend(self.learning_agents)
        agents.extend(list(self.other_policies.keys()))
        agent_vals = []
        for agent in agents:
            agent_total = 0.0
            # subtracting the values to allow for the min function at the end(max throws an error in the LP)
            for ai,action in enumerate(problem.joint_action_list):
                agent_total -= policy[ai]*q_values[agent][curr_state][action]
            agent_vals.append(agent_total)
        # Negated so the LP solver will maximize instead of minimize this value
        return -cvxopt.modeling.min(agent_vals)
    
    def libertarian_Q(self,agent_name,q_values,policy,curr_state,problem):
        agent_total = 0.0
        for ai,action in enumerate(problem.joint_action_list):
            agent_total += policy[ai]*q_values[agent_name][curr_state][action]
        # Negated so the LP solver will maximize instead of minimize this value
        return -agent_total
        
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
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
                q_val = q_values[agent][curr_state][action]
                for indiv_action in indiv_actions:
                    altered_action[agent] = indiv_action
                    altered_q_val = q_values[agent][curr_state][altered_action]
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
        objective_value = self.objective_func(agent_name,q_values,policies,curr_state,problem)
        lp = op(objective_value,constraints)
        lp.solve()
        equilibrium = float(lp.objective.value()[0])
        q_del = self.lr*(joint_rewards[agent_name] + self.dr*equilibrium)
        return q_del
                    
                    