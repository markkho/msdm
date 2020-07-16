import matplotlib.pyplot as plt

from pyrlap.pyrlap2.core import \
    State, TaskVariable, Action, TERMINALSTATE, Multinomial, \
    TabularMarkovDecisionProcess, \
    ANDMarkovDecisionProcess

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
        xvar = TaskVariable("x", tuple(range(len(tileArray[0]))), ("state",))
        axvar = TaskVariable("ax", (-1, 0, 1), ("action",))
        yvar = TaskVariable("y", tuple(range(len(tileArray))), ("state",))
        ayvar = TaskVariable("ay", (-1, 0, 1), ("action",))
        super().__init__(variables=[xvar, axvar, yvar, ayvar])

        states = set([])
        xyfeatures = {}
        walls = set([])
        absorbingStates = set([])
        initstates = set([])
        statealiases = {}
        for y, row in enumerate(tileArray):
            y = len(tileArray) - y - 1
            for x, f in enumerate(row):
                s = State((xvar, yvar), (x, y))
                states.add(s)
                if f == TERMINALSTATE:
                    continue
                statealiases[(x, y)] = s

                if f in defaultFeatures:
                    continue
                if f in initFeatures:
                    initstates.add(s)
                if f in absorbingFeatures:
                    absorbingStates.add(s)
                if f in wallFeatures:
                    walls.add(s)
                xyfeatures[s] = f

        states.add(TERMINALSTATE)
        xyfeatures[TERMINALSTATE] = None
        self._states = sorted(states)
        self._statealiases = statealiases
        self._initstates = sorted(initstates)
        self._walls = sorted(walls)
        self._wallFeatures = wallFeatures
        self._absorbingStates = sorted(absorbingStates)

        actionaliases = {
            '^': Action((axvar, ayvar), (0, 1)),
            'v': Action((axvar, ayvar), (0, -1)),
            '>': Action((axvar, ayvar), (1, 0)),
            '<': Action((axvar, ayvar), (-1, 0)),
            'x': Action((axvar, ayvar), (0, 0)),
        }
        self._actionaliases = actionaliases
        self._actions = sorted(actionaliases.values())

        self._xyfeatures = xyfeatures
        self.successProb = successProb
        if featureRewards is None:
            featureRewards = {'g': 0}
        self._featureRewards = featureRewards
        self.stepCost = stepCost
        self.terminationProb = terminationProb

    @property
    def height(self):
        return max(self.getVar("y").domain) + 1

    @property
    def width(self):
        return max(self.getVar('x').domain) + 1

    @property
    def walls(self):
        return self._walls

    @property
    def wallFeatures(self):
        return self._wallFeatures

    @property
    def initStates(self):
        try:
            return self._initstates
        except AttributeError:
            pass
        s0 = self.getInitialStateDist()
        s0 = sorted([s for s in s0.support if s0.prob(s) > 0])
        self._initstates = s0
        return self._initstates

    @property
    def absorbingStates(self):
        """
        Absorbing states are those that lead to the terminal state always
        (excluding the terminal state).
        """
        try:
            return self._absorbingStates
        except AttributeError:
            pass
        abss = []
        for s in self.states:
            if s == TERMINALSTATE:
                continue
            is_abs = True
            for a in self.actions:
                if self.getNextStateDist(s, a).prob(TERMINALSTATE) != 1.0:
                    is_abs = False
            if is_abs:
                abss.append(s)
        self._absorbingStates = sorted(abss)
        return self._absorbingStates

    @property
    def locationFeatures(self):
        return self._xyfeatures

    def getNextStateDist(self, s, a) -> Multinomial:
        if s == TERMINALSTATE:
            return Multinomial([TERMINALSTATE, ])

        try:
            x, y = s.values
        except AttributeError:
            s = self._statealiases[s]
            x, y = s.values

        try:
            ax, ay = a.values
        except AttributeError:
            a = self._actionaliases[a]
            ax, ay = a.values

        nx, ny = x + ax, y + ay
        ns = State(s.variables, (nx, ny))

        termination = Multinomial([TERMINALSTATE, ])
        if s in self.absorbingStates:
            return termination

        if ns not in self.states:
            baseTransition = Multinomial([s, ])
        elif ns in self.walls:
            baseTransition = Multinomial([s, ])
        elif s == ns:
            baseTransition = Multinomial([s, ])
        else:
            baseTransition = \
                 Multinomial([s, ns],
                             probs=[1 - self.successProb, self.successProb])
        return baseTransition.mixWith(termination, 1 - self.terminationProb)

    def getReward(self, state, action, nextstate) -> float:
        if state == TERMINALSTATE:
            return 0.0
        f = self._xyfeatures.get(nextstate, None)
        return self._featureRewards.get(f, 0.0) + self.stepCost

    def getActionDist(self, state) -> Multinomial:
        return Multinomial([a for a in self._actions])

    def getInitialStateDist(self) -> Multinomial:
        return Multinomial([s for s in self._initstates])

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