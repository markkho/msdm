from pyrlap_old.pyrlap2.core.problemclasses.mdp.mdp import ANDMarkovDecisionProcess, \
    MarkovDecisionProcess

class FactoredMarkovDecisionProcess(MarkovDecisionProcess):
    # for another day...
    pass

class ANDFactoredMarkovDecisionProcess(ANDMarkovDecisionProcess,
                                       FactoredMarkovDecisionProcess):
    pass

