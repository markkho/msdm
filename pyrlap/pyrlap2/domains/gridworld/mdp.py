import matplotlib.pyplot as plt
import json
from typing import Iterable
from pyrlap.pyrlap2.core.utils.gridstringutils import  stringToElementArray

from pyrlap.pyrlap2.core.problemclasses.mdp import \
    TabularMarkovDecisionProcess, \
    ANDMarkovDecisionProcess
from pyrlap.pyrlap2.core.distributions import Multinomial
from pyrlap.pyrlap2.core.assignment import \
    AssignmentMap as Dict, AssignmentSet as Set

def dictToStr(d):
    return json.dumps(d, sort_keys=True)

TERMINALSTATE = {'x': -1, 'y': -1}
TERMINALDIST = Multinomial([TERMINALSTATE,])

class GridWorld(TabularMarkovDecisionProcess):
    def __init__(self,
                 tileArray,
                 tileArrayFormat=None,
                 featureRewards=None,
                 absorbingFeatures=("g",),
                 wallFeatures=("#",),
                 defaultFeatures=(".",),
                 initFeatures=("s",),
                 stepCost=-1,
                 successProb=1.0,
                 terminationProb=0.0
                 ):
        super().__init__()
        parseParams = {"colsep": "", "rowsep": "\n", "elementsep": "."}
        if not isinstance(tileArray, str):
            tileArray = "\n".join(tileArray)
        elementArray = stringToElementArray(tileArray, **parseParams)
        states = []
        walls = Set()
        absorbingStates = Set()
        initStates = Set()
        locFeatures = Dict()
        for y_, row in enumerate(elementArray):
            y = len(elementArray) - y_ - 1
            for x, elements in enumerate(row):
                s = {'x': x, 'y': y}
                states.append(s)
                if len(elements) > 0:
                    f = elements[0]
                    locFeatures[s] = f
                    if f in initFeatures:
                        # initStates[s] = None
                        initStates.add(s)
                    if f in absorbingFeatures:
                        # absorbingStates[s] = None
                        absorbingStates.add(s)
                    if f in wallFeatures:
                        # walls[s] = None
                        walls.add(s)
        states.append(TERMINALSTATE)
        actions = [
            {'dx': 0, 'dy': 0},
            {'dx': 1, 'dy': 0},
            {'dx': -1, 'dy': 0},
            {'dy': 1, 'dx': 0},
            {'dy': -1, 'dx': 0}
        ]
        self._actions = sorted(actions, key=dictToStr)
        self._states = sorted(states, key=dictToStr)
        self._initStates = sorted(list(initStates), key=dictToStr)
        self._absorbingStates = sorted(list(absorbingStates), key=dictToStr)
        self._walls = sorted(list(walls), key=dictToStr)
        self._locFeatures = locFeatures
        self.successProb = successProb
        if featureRewards is None:
            featureRewards = {'g': 0}
        self._featureRewards = featureRewards
        self.stepCost = stepCost
        self.terminationProb = terminationProb #basically discount rate
        self._height = len(elementArray)
        self._width = len(elementArray[0])


    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def walls(self):
        return list(self._walls)

    @property
    def wallFeatures(self):
        return self._wallFeatures

    @property
    def initStates(self):
        return list(self._initStates)

    @property
    def absorbingStates(self):
        """
        Absorbing states are those that lead to the terminal state always
        (excluding the terminal state).
        """
        return list(self._absorbingStates)

    @property
    def locationFeatures(self):
        return self._locFeatures

    def isTerminal(self, s):
        return s == TERMINALSTATE

    def getNextStateDist(self, s, a) -> Multinomial:
        if self.isTerminal(s):
            return TERMINALDIST
        if s in self.absorbingStates:
            return TERMINALDIST

        x, y = s['x'], s['y']
        ax, ay = a.get('dx', 0), a.get('dy', 0)
        nx, ny = x + ax, y + ay
        ns = {'x': nx, 'y': ny}

        if ns not in self.states:
            bdist = Multinomial([s,])
        elif ns in self.walls:
            bdist = Multinomial([s,])
        elif ns == s:
            bdist = Multinomial([s,])
        else:
            bdist = Multinomial(support=[s, ns], probs=[1 - self.successProb, self.successProb])
        
        return bdist*(1 - self.terminationProb) | TERMINALDIST*self.terminationProb

    def getReward(self, state, action, nextstate) -> float:
        if self.isTerminal(state) or self.isTerminal(nextstate):
            return 0.0
        f = self._locFeatures.get(nextstate, "")
        return self._featureRewards.get(f, 0.0) + self.stepCost

    def getActions(self, state) -> Iterable:
        if self.isTerminal(state):
            return [{'dx': 0, 'dy': 0}, ]
        return [a for a in self._actions]

    def getInitialStateDist(self) -> Multinomial:
        return Multinomial([s for s in self.initStates])

    def __and__(self, other):
        return ANDGridWorld(self, other)

    def plot(self,
             allElements=False,
             ax=None,
             figsize=None,
             figsizeMult=1,
             featureColors=None,
             plotWalls=True,
             plotInitStates=True,
             plotAbsorbingStates=True
             ):
        if allElements:
            plotInitStates = True
            plotAbsorbingStates = True
        from pyrlap.pyrlap2.domains.gridworld.plotting import GridWorldPlotter
        if featureColors is None:
            featureColors = {
                'g': 'yellow',
                'x': 'red',
            }
        if ax is None:
            if figsize is None:
                figsize = (self.width * figsizeMult,
                           self.height * figsizeMult)
            _, ax = plt.subplots(1, 1, figsize=figsize)

        gwp = GridWorldPlotter(gw=self, ax=ax)
        gwp.plotFeatures(featureColors=featureColors)
        if plotWalls:
            gwp.plotWalls()
        if plotInitStates:
            gwp.plotInitStates()
        if plotAbsorbingStates:
            gwp.plotAbsorbingStates()
        gwp.plotOuterBox()

        return gwp


class ANDGridWorld(ANDMarkovDecisionProcess, GridWorld):
    def __init__(self, mdp1, mdp2):
        assert (mdp1.height == mdp2.height) and (mdp1.width == mdp2.width)
        assert (mdp1.actions == mdp2.actions)
        ANDMarkovDecisionProcess.__init__(self, mdp1, mdp2)

        # this is mostly for plotting
        self._width = mdp1._width
        self._height = mdp1._height
        self._walls = sorted(list(Set(mdp1._walls) | Set(mdp2._walls)), key=dictToStr)
        self._initStates = sorted(list(Set(mdp1._initStates) | Set(mdp2._initStates)), key=dictToStr)
        self._absorbingStates = sorted(list(Set(mdp1._absorbingStates) | Set(mdp2._absorbingStates)), key=dictToStr)
        self._actions = sorted(list(Set(mdp1._actions) | Set(mdp2._actions)), key=dictToStr)
        self._states = sorted(list(Set(mdp1._states) | Set(mdp2._states)), key=dictToStr)

        #store as strings of features
        locFeatures = mdp1._locFeatures.merge(mdp2._locFeatures)
        locFeatures = [(k, mdp1._locFeatures.get(k, "")+mdp2._locFeatures.get(k, "")) for k, fs in locFeatures.items()]
        self._locFeatures = Dict(locFeatures)
