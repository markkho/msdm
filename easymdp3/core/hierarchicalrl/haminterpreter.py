from collections import namedtuple

from .ham import TERMINATION_ACTION

HAMState = namedtuple('HAMState', ['state', 'stack'])

# hierarchy of abstract machines interpreter
class HAMInterpreter(object):
    def __init__(self, mdp, ham):
        self.mdp = mdp
        self.ham = ham
        self.policies = ham.subpolicies

    def _validate_stack(self, state, stack):
        for pi, (pname, pparams) in enumerate(stack[:-1]):
            policy = self.policies[pname]
            valid_children = policy(state,
                                    stack[:pi],
                                    **dict(pparams))
            child = stack[pi + 1]
            if child not in valid_children:
                stack = stack[:pi + 1]
                break
        return stack

    def _next_stacks(self, state, stack):
        pname, pparams = stack[-1]
        if pname == TERMINATION_ACTION:
            return [stack[:-1], ]
        top_pol = self.policies[pname]
        next_calls = top_pol(state, stack, **dict(pparams))
        next_stacks = [stack + [nc, ] for nc in next_calls]
        return next_stacks

    def _next_choice_or_action(self, state, stack):
        ground_actions = self.mdp.available_actions(state)
        while True:
            next_stacks = self._next_stacks(state, stack)
            choices = [nst[-1] for nst in next_stacks]

            # machine needs to make a choice
            if len(next_stacks) > 1:
                return (stack, choices)
            # environment needs to make a choice
            if next_stacks[0][-1][0] in ground_actions:
                return (stack, choices)
            stack = next_stacks[0]

    # def _interpreter_loop(self,
    #                       init_state,
    #                       root_name,
    #                       root_params,
    #                       # i.e. the completion policy
    #                       agent_policy=None,
    #                       # i.e. how agent learns
    #                       agent_update=None,
    #                       choice_loop_steps=100
    #                       ):
    #     if agent_policy is None:
    #         agent_policy = lambda s, st, c: c[0]
    #     if agent_update is None:
    #         agent_update = lambda **kwargs: None
    #
    #     stack = [(root_name, root_params), ]
    #     state = init_state
    #     for _ in range(choice_loop_steps):
    #         stack, choices = self._next_choice_or_action(state, stack)
    #
    #         # make choice
    #         update_dict = {'state': state, 'stack': deepcopy(stack)}
    #         if len(choices) > 1:
    #             choice = agent_policy(state, stack, choices)
    #         else: #e.g. in case it returns a ground action and we need to trans
    #             choice = choices[0]
    #         update_dict['choice'] = choice
    #         stack.append(choice)
    #
    #         # check if ground action chosen
    #         ns, r, a = (None, None, None)
    #         g_acts = self.mdp.available_actions(state)
    #         if stack[-1][0] in g_acts:
    #             a = stack[-1][0]
    #             ns, r = self.mdp.transition_reward(state, a)
    #             stack = stack[:-1]
    #             state = ns
    #
    #             # validate stack
    #             stack = self._validate_stack(state, stack)
    #
    #         update_dict['next_state'] = ns
    #         update_dict['reward'] = r
    #         agent_update(**update_dict)

    #SMDP functions
    def get_init_state(self, root_name, root_params, init_state=None):
        stack = [(root_name, root_params),]
        if init_state is None:
            init_state = self.mdp.get_init_state()
        s = init_state

        #get the first joint state with a choice
        while True:
            stack, choices = self._next_choice_or_action(s, stack)
            if len(choices) == 1:
                ground_action = choices[0][0]
                ns = self.mdp.transition(s, ground_action)
                s = ns
                stack = self._validate_stack(s, stack)
            else:
                break

        return HAMState(state=s, stack=tuple(stack))

    def available_actions(self, state, stack):
        new_stack, choices = self._next_choice_or_action(state, list(stack))
        return choices

    def transition_timestep_reward(self, state, stack, a, discount_rate=.99):
        stack = list(stack)
        n_steps = 0
        reward = 0

        if a[0] == TERMINATION_ACTION:
            stack = stack[:-1]
            return HAMState(state=state, stack=tuple(stack)), n_steps, reward

        if a not in self.available_actions(state, stack):
            raise TypeError("Action is not available at choice point")

        stack.append(a)

        while True:
            g_acts = self.mdp.available_actions(state)
            if a[0] in g_acts:
                ga = a[0]
                ns, r = self.mdp.transition_reward(state, ga)
                stack = self._validate_stack(ns, stack)

                reward += r*discount_rate**n_steps
                n_steps += 1
                state = ns
                stack.pop()
            stack, choices = self._next_choice_or_action(state, stack)
            if len(choices) > 1:
                break
            else:
                a = choices[0][0]
        return HAMState(state=state, stack=tuple(stack)), n_steps, reward

    def get_abstract_state(self, state, stack):
        pname, pparams = stack[-1]
        policy = self.ham.subpolicies[pname]
        astate = policy.state_abstraction(state, stack, **dict(pparams))
        return astate
