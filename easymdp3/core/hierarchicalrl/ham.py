TERMINATION_ACTION = "terminate"

class HierarchyOfAbstractMachines(object):
    def __init__(self, subpolicies):
        self.subpolicies = {pn: P() for pn, P in subpolicies.items()}

class SubPolicy(object):
    def __init__(self):
        pass

    def is_terminal(self, s, stack, *args, **kwargs):
        return False

    def state_abstraction(self, s, stack, *args, **kwargs):
        #do this
        pass

    def call(self):
        raise NotImplementedError

    def __call__(self, s, stack, *args, **kwargs):
        if self.is_terminal(s, stack, *args, **kwargs):
            return [(TERMINATION_ACTION, ()), ]
        return self.call(s, stack, *args, **kwargs)