from collections import namedtuple

TERMINATION_ACTION = "terminate"

HAMState = namedtuple('HAMState', ['groundstate', 'stack'])

class HierarchyOfAbstractMachines(object):
    def __init__(self, mdp, abstract_machines):
        self.mdp = mdp
        self.abstract_machines = \
            {pn: P() for pn, P in abstract_machines.items()}

    # ============================================ #
    #   Methods for handling machine transitions   #
    # ============================================ #
    def _validate_stack(self, state, stack):
        """
        Checks that every child process can be validly called by their
        parent process.
        """
        for pi, (pname, pparams) in enumerate(stack):
            policy = self.abstract_machines[pname]
            valid_children = policy(state,
                                    stack[:pi],
                                    **dict(pparams))

            #check non-top policies
            if pi < (len(stack) - 1):
                child = stack[pi + 1]
                if child not in valid_children:
                    stack = stack[:pi + 1]
                    break

            # check the policy at the top
            else:
                if len(valid_children) == 1 and \
                        valid_children[0][0] == TERMINATION_ACTION:
                    stack = stack[:-1]
                    break
        return stack

    def _next_choices(self, s):
        """
        Returns the choices available at the current choice state s
        """
        if self.mdp.is_absorbing(s.groundstate) and len(s.stack) == 1:
            return [(self.mdp.TERMINAL_ACTION, ()), ]
        if self.mdp.is_terminal(s.groundstate):
            return [(self.mdp.TERMINAL_ACTION, ()), ]
        pname, pparams = s.stack[-1]
        policy = self.abstract_machines[pname]
        choices = policy(s.groundstate, list(s.stack), **dict(pparams))
        return choices

    # ======================= #
    #    MDP-like Interface   #
    # ======================= #
    def get_init_state(self, root_name='root', root_params=(),
                       init_ground_state=None):
        if init_ground_state is None:
            init_ground_state = self.mdp.get_init_state()
        gs = init_ground_state
        return HAMState(groundstate=gs, stack=((root_name, root_params), ))

    def available_actions(self, state):
        choices = self._next_choices(state)
        return choices

    def is_terminal(self, s):
        if self.mdp.is_terminal(s.groundstate) and len(s.stack) <= 1:
            return True
        return False

    def is_absorbing(self, s):
        if self.mdp.is_absorbing(s.groundstate) and len(s.stack) <= 1:
            return True
        return False

    # =========================== #
    #                             #
    #    HAM-specific Interface   #
    #                             #
    # =========================== #

    # ======================= #
    #       Test Methods      #
    # ======================= #
    def is_ground_action(self, a):
        aname, aparams = a
        if aname in self.mdp.available_actions():
            return True
        return False

    def is_termination_action(self, a):
        if a[0] == TERMINATION_ACTION:
            return True
        return False

    def subtask_validly_terminates(self, s):
        """
        Determines whether s.stack validly terminates in s
        """
        pname, pparams = s.stack[-1]
        policy = self.abstract_machines[pname]
        choices = policy(s.groundstate, s.stack, **dict(pparams))
        if (TERMINATION_ACTION, ()) in choices:
            return True
        return False

    # ======================= #
    #    Transition Methods   #
    # ======================= #
    def transition_timestep_reward(self, state, action):
        if action not in self.available_actions(state):
            raise TypeError("Action is not available at choice point")

        gs = state.groundstate
        stack = list(state.stack)

        ts = 0
        if self.is_ground_action(action):
            ga = action[0]
            ngs, r = self.mdp.transition_reward(gs, ga)
            ts += 1
        elif self.is_termination_action(action):
            ngs = gs
            r = 0
            stack = stack[:-1]
        else: #calling a child process
            ngs = gs
            r = 0
            stack.append(action)

        #validate if not at the terminal root state
        if not (self.mdp.is_terminal(ngs) and len(stack) == 1):
            stack = self._validate_stack(ngs, stack)

        ns = HAMState(groundstate=ngs, stack=tuple(stack))
        return ns, ts, r

    def transition_timestep_reward_dist(self, state, action):
        ns, ts, r = self.transition_timestep_reward(state, action)
        return {(ns, ts, r): 1}

    # ======================= #
    #    State Abstraction    #
    # ======================= #
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