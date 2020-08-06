import matplotlib.pyplot as plt
import json
from pyrlap.pyrlap2.domains.gridworld.parsing import stringToElementArray

from pyrlap.pyrlap2.core import \
    TabularMarkovDecisionProcess, \
    ANDMarkovDecisionProcess, AssignmentMap, Multinomial

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
        walls = AssignmentMap()
        absorbingStates = AssignmentMap()
        initStates = AssignmentMap()
        locFeatures = AssignmentMap()
        for y_, row in enumerate(elementArray):
            y = len(elementArray) - y_ - 1
            for x, elements in enumerate(row):
                if len(elements) > 0:
                    f = elements[0]
                else:
                    f = ''
                s = {'x': x, 'y': y}
                locFeatures[s] = f
                states.append(s)
                if f in initFeatures:
                    initStates[s] = None
                if f in absorbingFeatures:
                    absorbingStates[s] = None
                if f in wallFeatures:
                    walls[s] = None
        states.append(TERMINALSTATE)
        states = sorted(states, key=lambda s: json.dumps(s, sort_keys=True))

        actions = [
            {'dx': 0, 'dy': 0},
            {'dx': 1},
            {'dx': -1},
            {'dy': 1},
            {'dy': -1}
        ]
        actions = sorted(actions, key=lambda a: json.dumps(a, sort_keys=True))
        self._states = states
        self._initStates = initStates
        self._walls = walls
        self._wallFeatures = wallFeatures
        self._locFeatures = locFeatures
        self._absorbingStates = absorbingStates
        self._actions = actions
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

    def getActionDist(self, state) -> Multinomial:
        return Multinomial([a for a in self._actions])

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
                **{f: 'black' for f in self._wallFeatures}
            }
        if ax is None:
            if figsize is None:
                figsize = (self.width * figsizeMult,
                           self.height * figsizeMult)
            _, ax = plt.subplots(1, 1, figsize=figsize)

        gwp = GridWorldPlotter(gw=self, ax=ax)
        gwp.plotFeatures(featureColors=featureColors)
        if plotInitStates:
            gwp.plotInitStates()
        if plotAbsorbingStates:
            gwp.plotAbsorbingStates()

        return gwp


class ANDGridWorld(ANDMarkovDecisionProcess, GridWorld):
    def __init__(self, mdp1, mdp2):
        assert (mdp1.height == mdp2.height) and (mdp1.width == mdp2.width)
        ANDMarkovDecisionProcess.__init__(self, mdp1, mdp2)
        self._xyfeatures = {**mdp1._xyfeatures, **mdp2._xyfeatures}
        self._wallFeatures = tuple(sorted(set(mdp1._wallFeatures + mdp2._wallFeatures)))
        self._walls = sorted(set(mdp1._walls + mdp2._walls))
        self._actions = sorted(set(mdp1._actions + mdp2._actions))
        self._actionaliases = {**mdp1._actionaliases, **mdp2._actionaliases}