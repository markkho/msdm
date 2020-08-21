from pyrlap.core.problemclasses.mdp.mdp import ANDMarkovDecisionProcess, \
    MarkovDecisionProcess

class FactoredMarkovDecisionProcess(MarkovDecisionProcess):
    # for another day...
    pass

class ANDFactoredMarkovDecisionProcess(ANDMarkovDecisionProcess,
                                       FactoredMarkovDecisionProcess):
    pass

