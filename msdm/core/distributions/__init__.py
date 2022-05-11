from msdm.core.distributions.distributions import Distribution, FiniteDistribution
from msdm.core.distributions.discretefactortable import \
    DiscreteFactorTable
from msdm.core.distributions.dictdistribution import DictDistribution, DeterministicDistribution, UniformDistribution
from msdm.core.distributions.softmaxdistribution import SoftmaxDistribution

def flip(p=.5):
    return DictDistribution({False:1-p, True:p})

def uniform(support):
    return DictDistribution.uniform(support)
