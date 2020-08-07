
import numpy as np
from pyrlap.pyrlap2.core import TERMINALSTATE
from pyrlap.pyrlap2.domains import GridWorld
from pyrlap.pyrlap2.algorithms import VectorizedValueIteration

gw = GridWorld(['....g'])
gw.transitionmatrix
gw.rewardmatrix
gw.initialstatevec
gw.