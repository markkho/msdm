from typing import Hashable
from numbers import Rational

class State(Hashable): pass
class Action(Hashable): pass
class Reward(Rational): pass
class Observation(Hashable): pass
class Probability(Rational): pass