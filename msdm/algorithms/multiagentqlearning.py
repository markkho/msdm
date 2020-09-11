from msdm.core.algorithmclasses import Learns, Result
from msdm.core.problemclasses.stochasticgame import StochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from typing import Iterable
import numpy as np 

class MultiAgentQLearning(Learns):
    
    def __init__(self,num_episodes=200,learning_rate=.9,discount_rate=1.0,epsilon=0.0,default_q_value=0.0):
        self.num_episodes = num_episodes
        self.dr = discount_rate
        self._policy = None
        self.eps = epsilon
        self.lr = learning_rate
        self.res = Result()
        self.default_q_value = default_q_value
    
    def train_on(self, problem: StochasticGame,agent_names: Iterable) -> Result:
        """
        TODOS: 
        Decide on what default policy should be for other agents. Currently all agents 
        use q-learning, but could just have one agent update using q learning and then 
        pass in policies for other agents, or something in between(e.g. 2 of 5 agents using q-learning, 
        others have a different policy)
        """
        # initialize Q values for each agent using q learning 
        self.res.Q = {agent_name: AssignmentMap() for agent_name in agent_names}
        
        for state in problem.state_list:
            for agent_name in agent_names:
                self.res.Q[agent_name][state] = AssignmentMap()
                for action in problem.joint_action_dist(state).support:
                    self.res.Q[agent_name][state][action[agent_name]] = self.default_q_value        

        for i in range(self.num_episodes):
            curr_state = problem.initial_state_dist().sample()
            while not problem.is_terminal(curr_state):
                actions = {agent_name: None for agent_name in agent_names}
                for agent_name in agent_names:
                    indiv_actions = [act_dict[agent_name] for act_dict in problem.joint_action_dist(curr_state).support]
                    max_act = max(self.res.Q[agent_name][curr_state].items(),key=lambda k:k[1])[0]
                    # Choose action using epsilon-greedy policy 
                    action = np.random.choice([max_act,np.random.choice(indiv_actions)],p=[1-self.eps,self.eps])
                    actions[agent_name] = action
                nxt_st = problem.next_state_dist(curr_state,actions).sample()
                jr = problem.joint_rewards(curr_state,actions,nxt_st)
                # update q values for each agent 
                for agent_name in agent_names:
                    q_del = self.lr*(jr[agent_name] + \
                    self.dr*max(self.res.Q[agent_name][nxt_st].items(),key=lambda k:k[1])[1] \
                    - self.res.Q[agent_name][curr_state][actions[agent_name]])
                    self.res.Q[agent_name][curr_state][actions[agent_name]] += q_del
                curr_state = nxt_st
#         pi = np.log(np.zeros_like(q))
#         pi[q == np.max(q, axis=-1, keepdims=True)] = 1
#         pi = softmax(pi, axis=-1)

#         # create result object
        
#         res.problem = problem
#         res.policy = {}
#         res.policy = res.pi = TabularPolicy(mdp.state_list, mdp.action_list, policymatrix=pi)
#         res._valuevec = v
#         vf = AssignmentMap([(s, vi) for s, vi in zip(mdp.state_list, v)])
#         res.valuefunc = res.V = vf
#         res._qvaluemat = q
#         qf = AssignmentMap()
#         for si, s in enumerate(mdp.state_list):
#             qf[s] = AssignmentMap()
#             for ai, a in enumerate(mdp.action_list):
#                 qf[s][a] = q[si, ai]
#         res.actionvaluefunc = res.Q = qf
        return self.res
    