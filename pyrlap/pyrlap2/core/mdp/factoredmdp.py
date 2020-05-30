from pyrlap.pyrlap2.core.mdp.mdp import ANDMarkovDecisionProcess, \
    MarkovDecisionProcess

class FactoredMarkovDecisionProcess(MarkovDecisionProcess):
    # for another day...
    pass

class ANDFactoredMarkovDecisionProcess(ANDMarkovDecisionProcess,
                                       FactoredMarkovDecisionProcess):
    pass

