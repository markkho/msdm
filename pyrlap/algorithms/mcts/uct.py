from __future__ import division

import random
import numpy as np


class UpperConfidenceTreeSearch(object):
    def __init__(self, mdp, rollout_policy=None,
                 exploration=.1, discount_rate=1):
        self.mdp = mdp
        self.rollout_policy = rollout_policy
        self.exploration = exploration
        self.discount_rate = discount_rate
        self.start_node = None

    def solve(
            self, init_state=None,
            iterations=1000, print_rollout=False):
        if init_state is None:
            init_state = self.mdp.get_init_state()

        start_node = self.create_node(init_state)
        for i in xrange(iterations):
            node = self.treepolicy(start_node)
            payoff = self.rollout(node, print_rollout=print_rollout)
            self.backup(node, payoff)
        self.start_node = start_node

    def create_node(self, state, incoming_action=None, parent=None):
        if (incoming_action is not None) and (parent is not None):
            incoming_reward = self.mdp.reward(
                s=parent['state'],
                a=incoming_action,
                ns=state)
        else:
            incoming_reward = 0
        node = {
            'untried_actions': self.mdp.available_actions(state),
            'state': state,
            'children': [],
            'parent': parent,
            'is_terminal': self.mdp.is_terminal(state),
            'incoming_action': incoming_action,
            'visit_count': 0,
            'payoff_sum': 0,
            'max_payoff': -np.inf,
            'incoming_reward': incoming_reward
        }
        return node

    def treepolicy(self, node):
        while not node['is_terminal']:
            if len(node['untried_actions']) > 0:
                return self.expand(node)
            else:
                node = self.uct_bestchild(node)
        return node

    def expand(self, node):
        a_i = random.randrange(len(node['untried_actions']))
        a = node['untried_actions'].pop(a_i)
        s = node['state']
        ns = self.mdp.transition(s, a)
        new_node = self.create_node(state=ns, incoming_action=a, parent=node)
        node['children'].append(new_node)
        return new_node

    def uct_bestchild(self, node, exploration=None):
        if exploration is None:
            exploration = self.exploration
        max_children = []
        max_val = -np.inf
        node_vc = node['visit_count']
        for child in node['children']:
            future_val = child['payoff_sum'] / child['visit_count']
            val = child['incoming_reward'] + self.discount_rate * future_val
            child_vc = child['visit_count']
            exp_val = exploration * np.sqrt(2 * np.log(node_vc) / child_vc)
            uct_val = val + exp_val
            if uct_val > max_val:
                max_val = uct_val
                max_children = [child, ]
            elif uct_val == max_val:
                max_children.append(child)
        return np.random.choice(max_children)

    def backup(self, node, payoff):
        discount = 1
        while node is not None:
            # update the visit count and payoff sum for current node
            node['visit_count'] += 1
            node['payoff_sum'] += payoff
            node['max_payoff'] = max(node['max_payoff'], payoff)

            # increase the discount for the previous step
            discount *= self.discount_rate
            # add the incoming reward to the payoff for the parent backup
            payoff = node['incoming_reward'] + (payoff * discount)

            node = node['parent']

    def rollout(self, node, max_depth=2000, print_rollout=False):
        s = node['parent']['state']
        # a = node['incoming_action']
        ns = node['state']
        # r = node['incoming_reward']
        payoff = 0
        i = 0
        discount = 1
        traj = []
        while (not self.mdp.is_terminal(s)) and (i < max_depth):
            s = ns
            if self.rollout_policy is None:
                a = np.random.choice(self.mdp.available_actions(s))
            else:
                a = self.rollout_policy.get_action(s)
            ns = self.mdp.transition(s, a)
            r = self.mdp.reward(s=s, a=a, ns=ns)
            payoff += r * discount
            discount *= self.discount_rate
            if print_rollout:
                traj.append((s, a, ns, r))
            i += 1

        if print_rollout:
            print traj

        return payoff

    #

    #
    def run_from_start(self):
        n = self.start_node
        traj = []
        while len(n['children']) > 0:
            n = self.uct_bestchild(n, 0)
            a = n['incoming_action']
            traj.append((n['state'], a))
        return traj
