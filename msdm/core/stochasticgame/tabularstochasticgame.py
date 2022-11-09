import warnings
warnings.warn("Multi-agent domains/algorithms are still being tested/developed - use with caution!")
import json, logging
import numpy as np
from itertools import product
from functools import reduce
from tqdm import tqdm
from msdm.core.stochasticgame import StochasticGame
from msdm.core.assignment.assignmentset import AssignmentSet as Set
from msdm.core.distributions import DiscreteFactorTable as Pr
logger = logging.getLogger(__name__)

class TabularStochasticGame(StochasticGame):

    def __init__(self,agent_names,memoize=False):
        super(TabularStochasticGame,self).__init__(agent_names=agent_names,memoize=memoize)

    @property
    def state_list(self):
        try:
            return self._states
        except AttributeError:
            pass
        logger.info("State space unspecified; performing reachability analysis.")
        self._states = \
            sorted(self.reachable_states(),
                key=lambda d: json.dumps(d, sort_keys=True) if isinstance(d, dict) else d
            )
        return self._states

    @property
    def position_list(self):
        try:
            return self._positions
        except AttributeError:
            positions = Set()
            for state in self.state_list:
                for agent in state:
                    if state[agent] != True:
                        positions.add((state[agent]["x"],state[agent]["y"]))
            self._positions = list(positions)
            return self._positions


    @property
    def joint_action_list(self):
        try:
            return self._joint_actions
        except AttributeError:
            pass
        logger.info("Action space unspecified; performing reachability analysis.")
        actions = Set()
        for s in self.state_list:
            joint_actions = self.joint_actions(s)
            ja_keys,ja_values = zip(*joint_actions.items())
            all_joint_actions = [dict(zip(ja_keys, list(v))) for v in product(*ja_values)]
            for action in all_joint_actions:
                actions.add(action)
        self._joint_actions = sorted(actions,
                key=lambda d: json.dumps(d, sort_keys=True) if isinstance(d, dict) else d
            )
        return self._joint_actions

    @property
    def transitionmatrix(self):
        try:
            return self._tfmatrix
        except AttributeError:
            pass
        import sparse
        ss = self.state_list
        aa = self.joint_action_list
        scoords = []
        acoords = []
        nscoords = []
        probs = []
        for si, s in enumerate(tqdm(ss,desc="Generating Sparse Transition Matrix")):
            for ai, a in enumerate(aa):
                nsdist = self.next_state_dist(s, a)
                for nsi, ns in enumerate(nsdist.keys()):
                    # Getting state index from state_list
                    index = self.state_list.index(ns)
                    prob = nsdist.prob(ns)
                    if prob != 0.0:
                        scoords.append(si)
                        acoords.append(ai)
                        nscoords.append(index)
                        probs.append(prob)
        coords = []
        coords.append(scoords)
        coords.append(acoords)
        coords.append(nscoords)
        tf = sparse.COO(coords,probs,shape=(len(ss),len(aa),len(ss)))
        self._tfmatrix = tf
        return self._tfmatrix

    @property
    def actionmatrix(self):
        try:
            return self._actmatrix
        except AttributeError:
            pass
        ss = self.state_list
        aa = self.joint_action_list
        am = np.zeros((len(ss), len(aa)))
        for (si, ai), _ in np.ndenumerate(am):
            s, a = ss[si], aa[ai]
            if a in self.joint_action_dist(s).support:
                p = 1
            else:
                p = 0
            am[si, ai] = p
        self._actmatrix = am
        return self._actmatrix


    @property
    def rewardmatrix(self):
        try:
            return self._rfmatrix
        except AttributeError:
            pass
        ss = self.state_list
        aa = self.joint_action_list
        rf = np.zeros((len(ss), len(aa), len(ss),len(self.agent_names)))
        for si, s in enumerate(ss):
            for ai, a in enumerate(aa):
                nsdist = self.next_state_dist(s, a)
                for nsi, ns in enumerate(ss):
                    if ns not in nsdist.support:
                        continue
                    r = self.joint_rewards(s, a, ns)
                    rf[si, ai, nsi] = np.array([r[name] for name in self.agent_names])
        self._rfmatrix = rf
        return self._rfmatrix

    @property
    def stateactionrewardmatrix(self):
        try:
            return self._sarfmatrix
        except AttributeError:
            pass
        rf = self.rewardmatrix
        tf = self.transitionmatrix
        self._sarfmatrix = np.einsum("sant,san->sat", rf, tf)
        return self._sarfmatrix

    @property
    def initialstatevec(self):
        try:
            return self._s0vec
        except AttributeError:
            pass
        s0 = self.initial_state_dist()
        self._s0vec = np.array([s0.prob(s) for s in self.state_list])
        return self._s0vec

    @property
    def nonterminalstatevec(self):
        try:
            return self._ntvec
        except AttributeError:
            pass
        ss = self.state_list
        self._ntvec = np.array([0 if self.is_terminal(s) else 1 for s in ss])
        return self._ntvec

    @property
    def reachablestatevec(self):
        try:
            return self._reachablevec
        except AttributeError:
            pass
        reachable = self.reachable_states()
        self._reachablevec = np.array \
            ([1 if s in reachable else 0 for s in self.state_list])
        return self._reachablevec

    @property
    def absorbingstatevec(self):
        try:
            return self._absorbingstatevec
        except AttributeError:
            pass
        def is_absorbing(s):
            actions = self.joint_action_dist(s).support
            for a in actions:
                nextstates = self.next_state_dist(s, a).support
                for ns in nextstates:
                    if not self.is_terminal(ns):
                        return False
            return True
        self._absorbingstatevec = np.array([is_absorbing(s) for s in self.state_list])
        return self._absorbingstatevec

    def reachable_states(self, MAX_STATES=float('inf')):
        S0 = self.initial_state_dist().support
        frontier = Set(S0)
        visited = Set(S0)
        while len(frontier) > 0:
            if len(visited) > MAX_STATES:
                break
            s = frontier.pop()
            if self.is_terminal(s):
                continue
            else:
                actions = self.joint_actions(s)
                adists = []
                for agent in actions:
                    adist = Pr([{agent: action} for action in actions[agent]])
                    adists.append(adist)
                total_dist = reduce(lambda a, b: a & b, adists)
                for ja in total_dist.support:
                    for ns in self.next_state_dist(s, ja).support:
                        if ns not in visited:
                            frontier.add(ns)
                        visited.add(ns)
        return visited
