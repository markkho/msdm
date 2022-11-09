from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable
import warnings
import random
import numpy as np

from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.mdp import MarkovDecisionProcess, HashableState, FunctionalPolicy
from msdm.core.distributions.dictdistribution import DeterministicDistribution, DictDistribution
from msdm.core.exceptions import SpecificationException, AlgorithmException

class LAOStar(Plans):
    def __init__(
        self,
        heuristic : Callable[[HashableState], float],
        max_lao_star_iterations=int(1e5),
        dynamic_programming_iterations=100,
        randomize_action_order : bool=True,
        randomize_nextstate_order : bool=True,
        event_listener_class : "LAOStarEventListener" = None,
        seed=None,
    ):
        """
        LAO* is a heuristic search algorithm that works on MDPs.
        This algorithm converges to the optimal policy when
        provided with an admissible heuristic (one that
        never under-estimates the value at all states).

        Parameters
        ----------
        heuristic : Callable[[HashableState], float]
            A heuristic state value function

        max_lao_star_iterations: int
            Maximum number of iterations of the main LAO* loop

        dynamic_programming_iterations : int
            Maximum number of policy iteration iterations in
            the value revision step.

        randomize_action_order : bool
            If set to True, then the order of actions in a
            state node are randomized when the node is first
            visited (and then fixed thereafter). Otherwise,
            the order is based on what is returned by the MDP.

        randomize_nextstate_order : bool

        event_listener_class : LAOStarEventListener
            An LAO* event listener class

        seed : int
            Random seed

        References
        ----------
        Hansen, E. A., & Zilberstein, S. (2001). LAO*:
        A heuristic search algorithm that finds solutions
        with loops. Artificial Intelligence, 129(1-2), 35-62.
        """
        if not callable(heuristic):
            heuristic = (lambda heuristic: lambda s: heuristic)(heuristic)
        self.heuristic = heuristic
        self.max_lao_star_iterations = max_lao_star_iterations
        self.dynamic_programming_iterations = dynamic_programming_iterations
        self.randomize_action_order = randomize_action_order
        self.randomize_nextstate_order = randomize_nextstate_order
        self.seed = seed
        self.event_listener_class = event_listener_class

    def plan_on(self, mdp: MarkovDecisionProcess) -> PlanningResult:
        if self.event_listener_class is not None:
            self._event_listener = self.event_listener_class()
        else:
            self._event_listener = None
        explicit_graph, iterations = self._run_lao_star(mdp)
        solution_graph = explicit_graph.solution_graph()
        if not solution_graph.is_solved():
            warnings.warn(f"LAO* not converged after {self.max_lao_star_iterations} iterations")

        return PlanningResult(
            policy=self._create_policy(solution_graph, mdp),
            explicit_graph=explicit_graph,
            converged=solution_graph.is_solved(),
            solution_graph=solution_graph,
            iterations=iterations,
            state_value_map=explicit_graph.state_value_map(),
            initial_value=explicit_graph.initial_value(),
            event_listener=self._event_listener
        )

    def _run_lao_star(self, mdp):
        rng = random.Random(self.seed)
        explicit_graph = ExplicitStateGraph(
            mdp=mdp,
            heuristic=self.heuristic,
            randomize_action_order=self.randomize_action_order,
            randomize_nextstate_order=self.randomize_nextstate_order,
            rng=rng,
            dynamic_programming_iterations=self.dynamic_programming_iterations,
        )
        for i in range(self.max_lao_star_iterations):
            solution_graph = explicit_graph.solution_graph()
            if solution_graph.is_solved():
                break
            expand_states = [solution_graph.best_breadth_first_tip_state(), ]
            for s in expand_states:
                explicit_graph.expand_at(s)
            ancestors = explicit_graph.revise_value_from(expand_states)

            if self._event_listener:
                self._event_listener.main_lao_star_loop(locals())
        return explicit_graph, i

    def _create_policy(self, solution_graph, mdp):
        solution_policy = dict()
        for s, n in solution_graph.states_to_nodes.items():
            solution_policy[s] = DeterministicDistribution(n.optimal_action)
        @FunctionalPolicy
        @lru_cache(maxsize=None)
        def policy(s):
            try:
                return solution_policy[s]
            except KeyError:
                pass
            max_actions = []
            max_val = float('-inf')
            for a in mdp.actions(s):
                val = mdp.next_state_dist(s, a).expectation(
                    lambda ns : mdp.reward(s, a, ns) + mdp.discount_rate*self.heuristic(ns)
                ) 
                if val > max_val:
                    max_actions = [a]
                elif val == max_val:
                    max_actions.append(a)
                max_val = max(val, max_val)
            return DictDistribution.uniform(max_actions)
        return policy

class LAOStarEventListener(ABC):
    @abstractmethod
    def main_lao_star_loop(self, localvars):
        pass

class ExplicitStateGraph:
    def __init__(
        self,
        mdp : MarkovDecisionProcess,
        heuristic,
        randomize_action_order,
        randomize_nextstate_order,
        rng=random,
        dynamic_programming_iterations=100,
    ):
        self.mdp = mdp
        self.heuristic = heuristic
        self.rng = rng
        self.n_expanded = 0

        self.dynamic_programming_iterations = dynamic_programming_iterations
        self.randomize_action_order = randomize_action_order
        self.randomize_nextstate_order = randomize_nextstate_order

        self.states_to_nodes = {}
        self.initial_states = sorted(mdp.initial_state_dist().support, key=lambda s: self.rng.random())
        for s in self.initial_states:
            self._initialize_node(s)

        self.VALUE_DECIMAL_PRECISION = 10

    def states_by_visitorder(self):
        states = list(self.states_to_nodes.keys())
        order = sorted(states, key=lambda s: self.states_to_nodes[s]['visitorder'])
        return order

    def states_by_expandedorder(self):
        states = [s for s, n in self.states_to_nodes.items() if n['expanded']]
        order = sorted(states, key=lambda s: self.states_to_nodes[s]['expandedorder'])
        return order

    def solution_graph(self):
        """Return solution graph from initial states"""
        return SolutionGraph(self)

    def initial_value(self):
        v = 0
        for s, p in self.mdp.initial_state_dist().items():
            v += self.states_to_nodes[s].value*p
        return v

    def state_value_map(self):
        v = {}
        for n in self.states_to_nodes.values():
            v[n.state] = n.value
        return v

    def expand_while(self, condition):
        while True:
            unexpanded = [s for s, n in self.states_to_nodes.items() if not n.expanded]
            unexpanded = [s for s in unexpanded if not self.mdp.is_absorbing(s)]
            unexpanded = [s for s in unexpanded if condition(s)]
            if len(unexpanded) == 0:
                break
            for s in unexpanded:
                self.expand_at(s)

    def expand_at(self, state):
        """Expand the explicit graph at a non-terminal node"""
        node = self.states_to_nodes[state]
        assert node.state in self.states_to_nodes, "Only visited states can be expanded"
        assert id(self.states_to_nodes[node.state]) == id(node), "Node needs to be inside the explicit graph"
        assert not node.expanded
        assert node.expandedorder == -1
        node.expanded = True
        node.expandedorder = self.n_expanded
        self.n_expanded += 1
        s = state
        for a in node.action_order:
            if len(node.action_nextstates[a]) != 0:
                if len(set(node.action_order)) != len(node.action_order):
                    raise SpecificationException(f"Duplicate actions in state {s}: {node.action_order}")
                raise AlgorithmException("Unexpanded nodes should not have next states explored")
            action_nextstates = list(self.mdp.next_state_dist(s, a).support)
            if self.randomize_nextstate_order:
                action_nextstates = sorted(action_nextstates, key = lambda _ : self.rng.random())
            for ns in action_nextstates:
                if ns not in self.states_to_nodes:
                    nextnode = self._initialize_node(ns, parent_node=node)
                else:
                    nextnode = self.states_to_nodes[ns]
                    nextnode.parent_states.add(node.state)
                node.action_nextstates[a].append(nextnode.state)

    def revise_value_from(self, states):
        """
        Perform a dynamic programming update from states
        through their shared ancestors.
        """
        ancestors = {}
        for state in states:
            node = self.states_to_nodes[state]
            self.update_ancestors_of(node=node, ancestors=ancestors)
        ancestor_list = list(ancestors.values())
        self.dynamic_programming(ancestor_list)
        return ancestors

    def update_ancestors_of(self, node, ancestors):
        """
        Return all ancestors of a node. An ancestor is any node
        connected by valid parents.

        A valid parent of a node N is a node whose *optimal action*
        has N as as a successor node in its child list.
        """
        if ancestors is None:
            ancestors = {}
        frontier = [node,]
        while frontier:
            n = frontier.pop()
            ancestors[n.state] = n
            for parent_state in n.parent_states:
                if parent_state in ancestors:
                    continue
                parent_node = self.states_to_nodes[parent_state]
                parent_nextstates = parent_node.action_nextstates[parent_node.optimal_action]
                if n.state in parent_nextstates:
                    frontier.append(parent_node)

    def dynamic_programming(self, nodes):
        """Perform dynamic programming updates over a set of nodes"""
        dp_action_order = list(set.union(*[set(n.action_order) for n in nodes]))
        tf, rf, am = self._state_nodes_to_matrices(nodes, dp_action_order)
        pi, v, q = self._policy_iteration(tf, rf, am)

        for si, node in enumerate(nodes):
            assert id(self.states_to_nodes[node.state]) == id(node)
            assert np.isclose(pi[si, :].sum(), 1)
            action_vals = {a: v for a, v in zip(dp_action_order, q[si, :])}
            optimal_action = max(node.action_order, key=lambda a: action_vals[a])
            node.optimal_action = optimal_action
            node.value = v[si]

    def _policy_iteration(self, tf, rf, am):
        pi = am/am.sum(-1, keepdims=True)
        assert (am.sum(-1) > 0).all()
        for i in range(self.dynamic_programming_iterations):
            s_rf = np.einsum("sa,san,san->s", pi, tf, rf) # state rewards
            mp = np.einsum("sa,san->sn", pi, tf) # policy markov chain
            v = np.linalg.solve(np.eye(tf.shape[0]) - self.mdp.discount_rate * mp, s_rf) # state value
            q = rf[:, :, :] + self.mdp.discount_rate * v[None, None, :]
            q = np.einsum("san,san->sa", tf, q)
            q = np.around(q, decimals=self.VALUE_DECIMAL_PRECISION)

            # Calculate the new policy, taking into account
            # the "infinite" cost of unavailable actions.
            new_pi = np.zeros_like(pi)
            np.put_along_axis(new_pi, (q + np.log(am)).argmax(axis=1)[:, None], values=1, axis=1)

            # Check convergence
            converged = (new_pi == pi).all()
            if converged:
                break
            pi = new_pi
        assert converged
        return pi, v, q

    def _state_nodes_to_matrices(self, state_nodes, actions):
        """
        Convert a collection of state nodes into valid
        transition and reward matrix representing the
        subgraph.
        """
        assert isinstance(state_nodes, list)
        n_states = len(state_nodes)
        n_actions = len(actions)
        state_index = {n.state: i for i, n in enumerate(state_nodes)}
        action_index = {a: i for i, a in enumerate(actions)}
        tf = np.zeros((n_states + 1, n_actions, n_states + 1))
        am = np.zeros((n_states + 1, n_actions))
        am[-1, :] = 1
        rf = np.zeros((n_states + 1, n_actions, n_states + 1))
        for node in state_nodes:
            s = node.state
            si = state_index[s]
            if self.mdp.is_absorbing(s):
                tf[si, :, -1] = 1
                am[si, :] = 1
                continue
            for a in self.mdp.actions(s):
                ai = action_index[a]
                am[si, ai] = 1
                for ns, prob in self.mdp.next_state_dist(s, a).items():
                    reward = self.mdp.reward(s, a, ns)
                    if ns in state_index:
                        nsi = state_index[ns]
                        tf[si, ai, nsi] = prob
                        rf[si, ai, nsi] = reward
                    else:
                        tf[si, ai, -1] += prob
                        # If the next state is a real terminal state, future expected value is 0.
                        # Otherwise, if the next state is simply not in the subgraph,
                        # we treat it as a pseudo-terminal and add in the next state value.
                        # Also, the pseudo-terminal reward is a probability weighted sum
                        # of the rewards and needs to be renormalized by the total
                        # probability of entering the pseudo-terminal state (see below).
                        if self.mdp.is_absorbing(ns):
                            rf[si, ai, -1] += prob*reward
                        else:
                            rf[si, ai, -1] += prob*(reward + self.mdp.discount_rate*self.states_to_nodes[ns].value)
                # Renormalize weighted and summed pseudo-terminal rewards
                # by dividing by the *total* probability of entering the pseudo-terminal state
                if tf[si, ai, -1] > 0:
                    rf[si, ai, -1] /= tf[si, ai, -1]

        assert np.isclose(tf[:-1].sum(-1)[am[:-1].astype(bool)], 1).all(), "Non-terminal valid action transitions need to sum to 1.0"
        assert (rf[-1, :, :] == 0).all(), "Terminal to terminal rewards must be 0"
        return tf, rf, am

    def _initialize_node(self, s, parent_node=None):
        assert s not in self.states_to_nodes
        if self.randomize_action_order:
            action_order = sorted(self.mdp.actions(s), key=lambda a : self.rng.random())
        else:
            action_order = self.mdp.actions(s)
        self.states_to_nodes[s] = Node(
            state=s,
            value=self.heuristic(s),
            action_order=action_order,
            optimal_action=action_order[0],
            expanded=False,
            expandedorder=-1,
            visitorder=len(self.states_to_nodes),
            parent_states=set([]) if parent_node is None else {parent_node.state},
            action_nextstates={a : [] for a in self.mdp.actions(s)}
        )
        return self.states_to_nodes[s]

class SolutionGraph:
    def __init__(self, explicit_graph):
        self.states_to_nodes = {}
        self.nonterminal_tip_states = []
        for s0 in explicit_graph.initial_states:
            frontier = [s0, ]
            while frontier:
                s = frontier.pop()
                if s in self.states_to_nodes:
                    continue
                node = explicit_graph.states_to_nodes[s]
                self.states_to_nodes[s] = node
                if node.expanded:
                    nextstates = node.action_nextstates[node.optimal_action]
                    frontier.extend(nextstates)
                else:
                    self.nonterminal_tip_states.append(node.state)

    def is_solved(self):
        return len(self.nonterminal_tip_states) == 0

    def best_breadth_first_tip_state(self):
        """
        Return the tip state with the highest value
        and was initialized earliest.
        """
        max_val = -float('inf')
        min_visit = float('inf')
        for s in self.nonterminal_tip_states:
            n = self.states_to_nodes[s]
            if n.value > max_val:
                best_node = n
                min_visit = n.visitorder
                max_val = n.value
            elif n.value == max_val and n.visitorder < min_visit:
                best_node = n
                min_visit = n.visitorder
                max_val = n.value
        assert best_node.value == max_val
        assert best_node.visitorder == min_visit
        return best_node.state

class Node(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
