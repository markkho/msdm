from collections import namedtuple

TERMINATION_ACTION = "terminate"

HAMState = namedtuple('HAMState', ['groundstate', 'stack'])

class HierarchyOfAbstractMachines(object):
    def __init__(self, mdp, abstract_machines):
        self.mdp = mdp
        self.abstract_machines = \
            {pn: P() for pn, P in abstract_machines.items()}

    def _next_choice_or_action(self, state, stack):
        """
            Execute policies until either:
                (1) we hit a choice point, or
                (2) we choose an MDP action
            Returns the stack at the point the choice needs to be
            made or when a ground action has been chosen
        """
        ground_actions = self.mdp.available_actions(state)
        while True:
            next_stacks = self._next_stacks(state, stack)
            choices = [nst[-1] for nst in next_stacks]
            # choices = self._next_stacks(state, stack)

            #environment needs to make a choice
            if len(choices) == 1:
                pname, pparams = choices[0]
                if pname in ground_actions:
                    return (stack, choices)
            #machine needs to make a choice
            else:
                return (stack, choices)

            stack = next_stacks[0]
            # stack.append(choices[0])

    def _next_stacks(self, state, stack):
        pname, pparams = stack[-1]
        if pname == TERMINATION_ACTION:
            return [stack[:-2], ]
        policy = self.abstract_machines[pname]
        next_calls = policy(state, stack, **dict(pparams))
        next_stacks = []
        for nc in next_calls:
            next_stacks.append(stack + [nc, ])
        # next_stacks = [stack + [nc, ] for nc in next_calls]
        return next_stacks

    def _validate_stack(self, state, stack):
        for pi, (pname, pparams) in enumerate(stack[:-1]):
            policy = self.abstract_machines[pname]
            valid_children = policy(state,
                                    stack[:pi],
                                    **dict(pparams))
            child = stack[pi + 1]
            if child not in valid_children:
                stack = stack[:pi + 1]
                break
        return stack

    # SMDP functions
    def get_init_state(self, root_name='root', root_params=(),
                       init_ground_state=None):
        stack = [(root_name, root_params), ]
        if init_ground_state is None:
            init_ground_state = self.mdp.get_init_state()
        gs = init_ground_state

        # get the first joint state with a choice
        while True:
            stack, choices = self._next_choice_or_action(gs, stack)
            if len(choices) == 1:
                ground_action = choices[0][0]
                ngs = self.mdp.transition(gs, ground_action)
                gs = ngs
                stack = self._validate_stack(gs, stack)
            else:
                break

        return HAMState(groundstate=gs, stack=tuple(stack))

    def available_actions(self, state):
        gs = state.groundstate
        stack = list(state.stack)
        _, choices = self._next_choice_or_action(gs, stack)
        return choices

    def is_ground_action(self, a):
        aname, aparams = a
        if aname in self.mdp.available_actions():
            return True
        return False

    def is_termination(self, a):
        if a[0] == TERMINATION_ACTION:
            return True
        return False

    def is_terminal(self, s):
        next_stacks = self._next_stacks(s.groundstate, list(s.stack))
        if len(next_stacks) == 1 and self.is_termination(next_stacks[0][-1]):
            return True
        return False

    def transition_type(self, s, a, ns=None):
        if a[0] == TERMINATION_ACTION:
            return "termination"
        if self.is_ground_action(a):
            return "ground"

        if ns is None:
            ns, _, _ = self.transition_timestep_reward(s, a)

        if ns == s:
            return "selfloop"

        if len(ns.stack) < len(s.stack):
            return "exit"

        if len(ns.stack) > len(s.stack):
            return "call"

        return "machine"

    def transition_timestep_reward(self, state, action, discount_rate=.99):
        gs = state.groundstate
        stack = list(state.stack)

        ts = 0
        reward = 0

        if action[0] == TERMINATION_ACTION:
            stack = stack[:-1]
            ns = HAMState(groundstate=gs, stack=tuple(stack))
            return ns, ts, reward

        if action not in self.available_actions(state):
            raise TypeError("Action is not available at choice point")

        stack.append(action)

        while True:
            g_acts = self.mdp.available_actions(gs)
            if action[0] in g_acts:
                ga = action[0]
                ns, r = self.mdp.transition_reward(gs, ga)
                stack = self._validate_stack(ns, stack)

                reward += r * discount_rate ** ts
                ts += 1
                gs = ns
                stack.pop()
            stack, choices = self._next_choice_or_action(gs, stack)
            if len(choices) > 1:
                break
            else:
                action = choices[0][0]
        ns = HAMState(groundstate=gs, stack=tuple(stack))
        return ns, ts, reward

    def transition_timestep_reward_dist(
            self, state, action, discount_rate=.99
    ):
        ns, ts, r = \
            self.transition_timestep_reward(state, action, discount_rate)
        return {(ns, ts, r): 1}

    def get_abstract_state(self, state):
        gs = state.groundstate
        stack = list(state.stack)

        pname, pparams = stack[-1]
        policy = self.abstract_machines[pname]
        astate = policy.state_abstraction(gs, stack, **dict(pparams))
        return astate



class AbstractMachine(object):
    def __init__(self):
        pass

    def is_terminal(self, s, stack, *args, **kwargs):
        return False

    def state_abstraction(self, s, stack, *args, **kwargs):
        pass

    def call(self):
        raise NotImplementedError

    def __call__(self, s, stack, *args, **kwargs):
        if self.is_terminal(s, stack, *args, **kwargs):
            return [(TERMINATION_ACTION, ()), ]
        return self.call(s, stack, *args, **kwargs)