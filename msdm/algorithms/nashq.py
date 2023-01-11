from msdm.core.algorithmclasses import Result
from msdm.algorithms.multiagentqlearning import TabularMultiAgentQLearner
from msdm.core.stochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from typing import Iterable
from tqdm import tqdm
import nashpy as nash
import itertools 
import numpy as np 
import warnings

"""
Use prebuilt software for computing nash equilibria?(Gambit can do N>2 players, but requires separate installation)
"""
# Use nashpy for now 
class NashQLearner(TabularMultiAgentQLearner):
    
    def __init__(self,learning_agents: Iterable,
                 other_policies:dict,num_episodes=200,
                 learning_rate=.1,discount_rate=1.0,
                 epsilon=0.0,epsilon_decay=1.0,default_q_value=0.0,
                 show_progress=False,alg_name="Nash Q-Learning",render=False,render_from=0):
        super().__init__(learning_agents,other_policies,num_episodes,
                        learning_rate,discount_rate,epsilon,epsilon_decay,
                        default_q_value,all_actions=True,
                        show_progress=show_progress,alg_name=alg_name,render=render,render_from=render_from)
    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        if problem.is_terminal(next_state):
            return self.lr*joint_rewards[agent_name]
        indiv_actions = {agent:problem.joint_actions(next_state)[agent] for agent in problem.joint_actions(next_state)}
        next_actions = problem.joint_actions(next_state)
        agent_one_actions = list(next_actions[agent_name])
        for agent in next_actions:
            if agent == agent_name:
                continue 
            agent_two = agent
            agent_two_actions = list(next_actions[agent])
        payoff_matrices = {agent: np.zeros((len(agent_one_actions),len(agent_two_actions))) for agent in next_actions}
        for ai,action in enumerate(agent_one_actions):
            for ai_two,action_two in enumerate(agent_two_actions):
                ja = {agent_name:action,agent_two:action_two}
                payoff_matrices[agent_name][ai][ai_two] = q_values[agent_name][next_state][ja]
                payoff_matrices[agent_two][ai][ai_two] = q_values[agent_two][next_state][ja]
        game = nash.Game(payoff_matrices[agent_name],payoff_matrices[agent_two])
        # Throws a runtime warning about degenerate games otherwise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eqs = list(game.support_enumeration())
        rand_eq = np.random.choice(len(eqs))
        rand_eq = eqs[rand_eq]
        q_val_updates = AssignmentMap()
        action_one_index = np.random.choice(len(rand_eq[0]),p=rand_eq[0])
        sample_action_one = agent_one_actions[action_one_index]
        action_two_index = np.random.choice(len(rand_eq[1]),p=rand_eq[1])
        sample_action_two = agent_two_actions[action_two_index]
        joint_action = {agent_name:sample_action_one,agent_two:sample_action_two}
        q_val = q_values[agent_name][next_state][joint_action]*rand_eq[0][action_one_index]*rand_eq[1][action_two_index]
        q_del = (joint_rewards[agent_name] + self.dr*q_val)
        return q_del