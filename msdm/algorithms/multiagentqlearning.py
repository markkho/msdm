from msdm.core.algorithmclasses import Result
from msdm.core.algorithmclasses import Learns
from msdm.core.stochasticgame.tabularstochasticgame import TabularStochasticGame
from msdm.core.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy, SingleAgentPolicy
from msdm.core.assignment.assignmentmap import AssignmentMap
from tqdm import tqdm
from typing import Iterable
import numpy as np 
import itertools 
from scipy.special import softmax
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time

    
class TabularMultiAgentQLearner(Learns):
    
    def __init__(self,learning_agents:Iterable,other_policies:dict,num_episodes=200,
                 learning_rate=.1,discount_rate=1.0,epsilon=0.0,
                 epsilon_decay=1.0,default_q_value=0.0,max_steps=50,all_actions=True,
                 show_progress=False,render=False,render_from=0,alg_name="Q-Learning"):
        self.learning_agents = learning_agents 
        self.other_agents = list(other_policies.keys())
        self.other_policies = other_policies 
        self.all_agents = [] 
        self.all_agents.extend(self.learning_agents)
        self.all_agents.extend(self.other_agents)
        self.num_episodes = num_episodes 
        self.lr = learning_rate 
        self.dr = discount_rate 
        self.eps = epsilon 
        self.default_q_value = default_q_value 
        self.show_progress = show_progress
        self.all_actions = all_actions 
        self.alg_name = alg_name
        self.epsilon_decay = epsilon_decay
        self.render = render
        self.render_from = render_from
        self.errors = []
        self.max_steps = max_steps
        # rendering animation and progress bars don't play nicely together in jupyter lab
        if self.render:
            self.show_progress = False

    
    
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
        if self.render: 
            agent_names = copy(self.learning_agents)
            agent_names.extend(self.other_agents)
            figure, axes = plt.subplots(1,2,figsize=(20,10),gridspec_kw={"width_ratios":[1,3]})
            figure.suptitle(self.alg_name)
            renderer = Renderer(problem,figure,axes[1],agent_names,self.all_actions,initial_epsilon=self.eps,gamma=self.dr,info_axis=axes[0])
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
                        
        # Adds a progress bar 
        if self.show_progress:
            episodes = tqdm(range(self.num_episodes),desc="Training with " + self.alg_name)
        else:
            episodes = range(self.num_episodes)
            
        for i in episodes:
            curr_state = problem.initial_state_dist().sample()
            curr_step = 0
            avg_error = 0.0
            if self.render and i >= self.render_from:
                all_agents = [agent for agent in self.learning_agents]
                all_agents.extend(self.other_agents)
                renderer.plotter.plot_new_state(curr_state)
                renderer.clear_func(wait=True)
                renderer.display_func(figure)
                time.sleep(renderer.interval)
            while not problem.is_terminal(curr_state) and curr_step < self.max_steps:
                # Choose action 
                actions = self.pick_action(curr_state,res.Q,problem)
                curr_state,actions,jr,nxt_st = self.step(problem,curr_state,actions)
                if self.render and i >= self.render_from:
                    if not problem.is_terminal(nxt_st):
                        renderer.render_frame(i,curr_step,res.Q,actions,jr,curr_state,nxt_st,self.eps)
                # update q values for each agent 
                for agent_name in self.learning_agents:
                    new_q = self.update(agent_name,actions,res.Q,jr,curr_state,nxt_st,problem)
                    if not self.all_actions:
                        old_q = res.Q[agent_name][curr_state][actions[agent_name]]
                        res.Q[agent_name][curr_state][actions[agent_name]] = (1-self.lr)*res.Q[agent_name][curr_state][actions[agent_name]] + self.lr*new_q
                        avg_error += abs(old_q - res.Q[agent_name][curr_state][actions[agent_name]])
                    else:
                        old_q = res.Q[agent_name][curr_state][actions]
                        res.Q[agent_name][curr_state][actions] = (1-self.lr)*res.Q[agent_name][curr_state][actions] + self.lr*new_q
                        avg_error += abs(old_q - res.Q[agent_name][curr_state][actions])
                curr_state = nxt_st
                curr_step += 1
            avg_error /= (curr_step*len(self.learning_agents)) 
            self.errors.append(avg_error)
            self.eps *= self.epsilon_decay

        # Converting to dictionary representation of deterministic policy
        pi = self.compute_deterministic_policy(res.Q,problem)

        # add in non_learning agents 
        for agent in self.other_agents:
            pi[agent] = self.other_policies[agent]
            
        # create result object
        res.problem = problem
        res.policy = {}
        res.policy = res.pi = TabularMultiAgentPolicy(problem, pi,self.dr,show_progress=self.show_progress)
        # Removing animation if present
        if self.render:
            plt.close()
        return res
    
    def plan_on(self,problem: TabularStochasticGame,delta=.0001):
        # initialize Q values for each agent using q learning
        res = Result()
        res.Q = {agent_name: AssignmentMap() for agent_name in self.all_agents}
        
        for state in problem.state_list:
            for agent_name in self.all_agents:
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
        # Currently does full sweeps of state-action space. Would be nice to add in a function 
        # as a parameter that determines order/priority of updates
        q_del = delta
        while q_del >= delta:
            q_del = 0.0
            agent_deltas = {agent:0.0 for agent in self.learning_agents}
            if self.show_progress:
                states = tqdm(problem.state_list)
            else:
                states = problem.state_list
            avg_error = 0.0
            for state in states:
                for action in problem.joint_action_list:
                    next_vals = {agent:0.0 for agent in self.learning_agents}
                    for next_state,prob in problem.next_state_dist(state,action).items(probs=True):
                        rewards = problem.joint_rewards(state,action,next_state)
                        for agent in self.learning_agents:
                            new_q = self.update(agent,action,res.Q,rewards,state,next_state,problem)
                            next_vals[agent] += prob*new_q
                    for agent in self.learning_agents:
                        old_q = res.Q[agent][state][action]
                        res.Q[agent][state][action] = next_vals[agent]
                        avg_error += abs(old_q - res.Q[agent][state][action])
                        if abs(old_q - res.Q[agent][state][action]) > q_del:
                            q_del = abs(old_q - res.Q[agent][state][action])
                        if abs(old_q - res.Q[agent][state][action]) > agent_deltas[agent]:
                            agent_deltas[agent] = abs(old_q - res.Q[agent][state][action])
            print(agent_deltas)
            self.errors.append(avg_error/(len(states)*len(problem.joint_action_list)*len(self.learning_agents)))
        print(f"Planning for {self.alg_name} Complete")
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
        pi = AssignmentMap()
        for agent in q_values:
            pi[agent] = AssignmentMap()
            for state in q_values[agent]:
                pi[agent][state] = AssignmentMap()
                # Picks randomly among maximum actions 
                max_val = max(q_values[agent][state].items(),key=lambda k:k[1])[1]
                max_acts = []                
                for act in problem.joint_action_list:
                    if self.all_actions:
                        if q_values[agent][state][act] == max_val:
                            max_acts.append(act)
                    else:
                        if q_values[agent][state][act[agent]] == max_val:
                            max_acts.append(act[agent])
                max_act = np.random.choice(max_acts)
                if self.all_actions:
                    max_act = max_act[agent]
                    
                for ai,action in enumerate(problem.joint_action_list):
                    if self.all_actions:
                        if action[agent] == max_act:
                            pi[agent][state][action[agent]] = 1.0
                        else:
                            pi[agent][state][action[agent]] = 0.0
                    else:
                        if action[agent] == max_act:
                            pi[agent][state][action[agent]] = 1.0
                        else:
                            pi[agent][state][action[agent]] = 0.0
            pi[agent] = SingleAgentPolicy(agent,problem,pi[agent],q_values[agent],self.all_actions)
        return pi 

    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        if problem.is_terminal(next_state):
            return joint_rewards[agent_name]
        if problem.is_terminal(curr_state):
            return 0.0
        q_del = joint_rewards[agent_name]
        q_del += self.dr*(max(q_values[agent_name][next_state].items(),key=lambda k:k[1])[1])
        return q_del 
    
class Renderer():
    
    def __init__(self,problem,figure,image_axis,agent_names,all_actions=True,initial_epsilon=0.0,gamma=1.0,info_axis=None,interval=.2):
        import IPython.display as display
        # Was getting a weird name conflict for display, not sure where it's happening yet
        self.clear_func = display.clear_output
        self.display_func = display.display
        self.interval = interval
        self.figure = figure 
        self.problem = problem
        self.image_axis = image_axis 
        self.info_axis = info_axis
        self.agents = agent_names
        self.epsilon = initial_epsilon
        self.gamma = gamma
        self.all_actions = all_actions
        self.keyword_objs, self.plotter = self.init_animation()
        display.display(self.figure)
    
    def render_frame(self,episode_num,step_num,q_values,actions,joint_rewards,curr_state,next_state,epsilon):
        self.plotter.plot_new_state(next_state)
        self.epsilon = epsilon
        if self.info_axis != None:
            self.update_info(episode_num,step_num,q_values,actions,joint_rewards,curr_state,next_state)
        self.clear_func(wait=True)
        self.display_func(self.figure)
        time.sleep(self.interval)
        return self
        
    def init_animation(self,featurecolors=None):
        from msdm.domains.gridgame.plotting import GridGamePlotter
        if featurecolors is None:
            
            featurecolors = {
                "fence": "brown",
                "obstacle": "black",
                "wall": "gray"
            }
        if self.info_axis != None:
            text_objs = self.init_info(self.figure)
            self.info_axis.axis("off")
        else:
            text_objs = None
        gwp = GridGamePlotter(gg=self.problem, ax=self.image_axis)
        gwp.plot_features(featurecolors=featurecolors)
        gwp.plot_walls()
        gwp.plot_initial_states()
        gwp.plot_absorbing_states()
        gwp.plot_fences()
        gwp.plot_outer_box()
        return text_objs,gwp
    
    def init_info(self,figure):
        keywords = {"Current Episode":0,"Current Step":0,"Sum of Total Rewards":0.0,"Epsilon":self.epsilon}
        agent_keywords = {"Total Reward":0.0,"Episodic Reward":0.0,"Current State Value":0.0,"Most Recent Action":"None","Temporal Difference":0.0}
        agent_height = .95/(len(self.agents)+1)
        keyword_space = agent_height/len(keywords)
        text_objs = AssignmentMap()
        for agent in self.agents:
            text_objs[agent] = {}
        text_objs["general"] = {}
        bbox_props = {"boxstyle":"round","facecolor":"wheat","alpha":0.5}
        for i,keyword in enumerate(keywords):
            text_objs["general"][keyword] = self.info_axis.annotate(keyword +f": {keywords[keyword]}",(0.0,.95-keyword_space*i),xycoords=self.info_axis,bbox=bbox_props)
        
        agent_keyword_space = agent_height/(len(agent_keywords) + 1)
        for k,agent in enumerate(self.agents):
            bbox_props = {"boxstyle":"round","facecolor":"wheat","alpha":0.5}
            text_objs[agent]["name"] = self.info_axis.annotate(agent,(0.0,.95 - agent_height*(k+1)),xycoords=self.info_axis,bbox=bbox_props)
            for i,agent_keyword in enumerate(agent_keywords):
                text_objs[agent][agent_keyword] = self.info_axis.annotate(agent_keyword + f": {agent_keywords[agent_keyword]}",(0.0,.95 - (agent_height*(k+1) + agent_keyword_space*(i+1))),xycoords=self.info_axis,bbox=bbox_props)
        return text_objs
    
    def update_info(self,episode_num,step_num,q_values,actions,joint_rewards,curr_state,next_state):
        for agent in self.keyword_objs:
            if agent == "general":
                for keyword in self.keyword_objs[agent]:
                    if keyword == "Current Episode":
                        self.keyword_objs[agent][keyword].set_text(f"Current Episode: {episode_num}")
                    elif keyword == "Current Step":
                        self.keyword_objs[agent][keyword].set_text(f"Current Step: {step_num}")
                    elif keyword == "Sum of Total Rewards":
                        curr_text = self.keyword_objs[agent][keyword].get_text()
                        value = float(curr_text.split()[-1])
                        value += sum(joint_rewards.values())
                        self.keyword_objs[agent][keyword].set_text(f"Sum of Total Rewards: {value}")
                    elif keyword == "Epsilon":
                        self.keyword_objs[agent][keyword].set_text(f"Epsilon: {self.epsilon}")
                    else:
                        raise Exception("Keyword not in general statistics tracked")
            else:
                for keyword in self.keyword_objs[agent]:
                    if keyword == "Total Reward":
                        curr_text = self.keyword_objs[agent][keyword].get_text()
                        value = float(curr_text.split()[-1])
                        value += joint_rewards[agent]
                        self.keyword_objs[agent][keyword].set_text(f"Total Reward: {value}")
                    elif keyword == "Episodic Reward":
                        if step_num == 0:
                            self.keyword_objs[agent][keyword].set_text(f"Episodic Reward: {0.0}")
                        else:
                            curr_text = self.keyword_objs[agent][keyword].get_text()
                            value = float(curr_text.split()[-1])
                            value += joint_rewards[agent]
                            self.keyword_objs[agent][keyword].set_text(f"Episodic Reward: {value}")
                    elif keyword == "Current State Value":
                        if agent not in q_values.keys():
                            self.keyword_objs[agent][keyword].set_text(f"Q-values not stored for {agent}")
                        else:
                            curr_vals = q_values[agent][next_state]
                            max_q = max(curr_vals.items(),key=lambda x: x[1])[1]
                            self.keyword_objs[agent][keyword].set_text(f"Current State Value: {max_q}")
                    elif keyword == "Most Recent Action":
                        action = actions[agent]
                        self.keyword_objs[agent][keyword].set_text(f"Most Recent Action: {action}")
                    elif keyword == "Temporal Difference":
                        if self.all_actions:
                            temporal_difference = q_values[agent][curr_state][actions]
                        else:
                            temporal_difference = q_values[agent][curr_state][actions[agent]]
                        next_step = joint_rewards[agent]
                        next_step += self.gamma*(max(q_values[agent][next_state].items(),key=lambda x: x[1])[1])
                        temporal_difference = next_step - temporal_difference 
                        self.keyword_objs[agent][keyword].set_text(f"Temporal Difference: {temporal_difference}")
                    elif keyword == "name":
                        continue
                    else:
                        raise Exception("Keyword not in agent statistics tracked")
        
    
    
    
        
    
