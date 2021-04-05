from types import SimpleNamespace
import warnings, tqdm
import random
import numpy as np

from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.problemclasses.mdp import MarkovDecisionProcess, TabularPolicy
from msdm.core.distributions.dictdistribution import DeterministicDistribution

class LAOStar(Plans):
    """
    Hansen, E. A., & Zilberstein, S. (2001). LAO*: 
    A heuristic search algorithm that finds solutions 
    with loops. Artificial Intelligence, 129(1-2), 35-62.
    """
    def __init__(self,
                 heuristic=None, # Function over states
                 egraph=None,
                 show_warning=False,
                 show_progress=True,
                 max_lao_iters=int(1e5),
                 policy_evaluation_iters=100,
                 policy_iteration_iters=100,
                 seed=None):
        if heuristic is None:
            heuristic = lambda s : 0.0
        A = SimpleNamespace(**{n: a for n, a in locals().items() if n != "self"})
        self.A = A

    def plan_on(self, mdp: MarkovDecisionProcess) -> PlanningResult:
        A = self.A
        if A.seed is None:
            seed = random.randint(1, 1e20)
        else:
            seed = A.seed
        random.seed(seed)
        
        discount_rate = mdp.discount_rate
        #initialize explicit graph
        if A.egraph is None:
            egraph = {} #explicit graph
        initStates = mdp.initial_state_dist().support
        for s0 in initStates:
            actionorder = list(mdp.actions(s0))
            random.shuffle(actionorder)
            node = {
                "parents": [], 
                "actionchildren": dict(),
                "state": s0, 
                "value": A.heuristic(s0),
                "bestaction": actionorder[0],
                "actionorder": actionorder,
                "visitorder": len(egraph),
                "expandedorder": -1,
                "expanded": False
            }
            egraph[s0] = node

        def policy_improvement(graph, egraph):
            pichange = False
            for n in graph.values():
                s = n["state"]
                aa = n["actionorder"]
                avals = []
                maxa = n["bestaction"]
                maxav = -np.inf
                for a in aa:
                    aval = 0
                    for ns, p in mdp.next_state_dist(s, a).items():
                        aval += p*(mdp.reward(s, a, ns) + discount_rate * egraph[ns]["value"])
                    if aval > maxav:
                        maxav = aval
                        maxa = a
                if n["bestaction"] != maxa:
                    pichange = True
                n["bestaction"] = maxa
            return pichange

        def policy_evaluation(graph, egraph):
            #NOTE: this can fail to converge if the current policy is stuck in a loop
            #TODO: do policy evaluation with matrix inversion / linear equation solving
            for iPE in range(A.policy_evaluation_iters):
                endPE = False
                valchange = -np.inf
                for n in graph.values():
                    assert n['expanded']
                    s, a = n["state"], n["bestaction"]
                    expval = 0
                    for ns, p in mdp.next_state_dist(s, a).items():
                        nextnode = egraph[ns]
                        expval += p*(mdp.reward(s, a, ns) + discount_rate * nextnode["value"])
                    valchange = np.max([np.abs(n["value"] - expval), valchange])
                    n["value"] = expval
                if valchange < 1e-6:
                    break
                if iPE == (A.policy_evaluation_iters - 1) and A.show_warning:
                    #Note: whenever backtracking occurs, this will fail to converge
                    warnings.warn(f"Policy evaluation did not converge after {A.policy_evaluation_iters} iterations")

        def update_dynamic_programming(graph, egraph):
            #run policy iteration on all the states in the subgraph graph
            # it is assumed that all of the nodes in graph are expanded
            policy_improvement(graph, egraph)
            for iDP in range(A.policy_iteration_iters):
                policy_evaluation(graph, egraph)
                pichange = policy_improvement(graph, egraph)
                if not pichange:
                    break

        def get_nonterminal_tips(sGraph):
            ntt = []
            for n in sGraph.values():
                if (not n['expanded']) and (not mdp.is_terminal(n['state'])):
                    ntt.append(n)
            return ntt

        def get_solution_graph(egraph, initStates):
            sGraph = {}
            for s0 in initStates:
                toget = [s0, ]
                while len(toget) > 0:
                    s = toget.pop()
                    if s in sGraph:
                        continue
                    n = egraph[s]
                    sGraph[s] = n
                    bestchildren = n['actionchildren'].get(n['bestaction'], [])
                    bestchildrenstates = [cn['state'] for cn in bestchildren]
                    toget.extend(bestchildrenstates)
            return sGraph

        def expand_graph(egraph, n, nExpanded):
            s = n['state']
            assert not mdp.is_terminal(s), "Only expand non-terminal states"
            n['expanded'] = True
            n['expandedorder'] = nExpanded
            aa = n['actionorder']
            for a in aa:
                children = []
                nextstates = list(mdp.next_state_dist(s, a).support)
                random.shuffle(nextstates)
                for ns in nextstates:
                    if ns not in egraph:
                        actionorder = list(mdp.actions(ns))
                        random.shuffle(actionorder)
                        nextnode = {
                            "state": ns,
                            "value": A.heuristic(ns),
                            "bestaction": actionorder[0],
                            "actionorder": actionorder,
                            "visitorder": len(egraph),
                            "expandedorder": -1,
                            "parents": [n, ],
                            "actionchildren": dict(),
                            "expanded": False
                        }
                        egraph[ns] = nextnode
                    else:
                        nextnode = egraph[ns]
                        nextnode['parents'].append(n)
                    children.append(nextnode)
                n['actionchildren'][a] = children

        def get_ancestors(egraph, tip):
            ans = {}
            toget = [tip,]
            while len(toget) > 0:
                n = toget.pop()
                hs = n['state']
                ans[hs] = n
                for pn in n['parents']:
                    if pn["state"] in ans:
                        continue
                    bestchildren = pn['actionchildren'][pn['bestaction']]
                    if n['state'] in [cn['state'] for cn in bestchildren]:
                        toget.append(pn)
            return ans


        if A.show_progress:
            pbar = tqdm.tqdm()
        nExpanded = 0
        for s0 in initStates:
            if mdp.is_terminal(s0):
                continue
            n0 = egraph[s0]
            expand_graph(egraph, n0, nExpanded)
            nExpanded += 1
            z = get_ancestors(egraph, n0)
            update_dynamic_programming(z, egraph)
        sGraph = get_solution_graph(egraph, initStates)
        for laoIter in range(A.max_lao_iters):
            if A.show_progress:
                pbar.update(1)
                pbar.set_description(f"|egraph|: {len(egraph)}; |sGraph| = {len(sGraph)}")
            
            ntt = get_nonterminal_tips(sGraph)
            if len(ntt) == 0:
                break
            nonterm = max(ntt, key=lambda n: (n["value"], -n["visitorder"]))
            expand_graph(egraph, nonterm, nExpanded)
            nExpanded += 1
            z = get_ancestors(egraph, nonterm)
            update_dynamic_programming(z, egraph)
            sGraph = get_solution_graph(egraph, initStates)

        if A.show_progress:
            pbar.close()

        pi = dict()
        for n in sGraph.values():
            pi[n['state']] = DeterministicDistribution(n['bestaction'])
        pi = TabularPolicy(pi)

        if laoIter == (A.max_lao_iters - 1):
            warnings.warn(f"LAO* not converged after {A.max_lao_iters} iterations")
            converged = False
        else:
            converged = True

        initial_value = sum([sGraph[s0]['value']*p for s0, p in mdp.initial_state_dist().items()])

        return PlanningResult(
            converged=converged,
            egraph=egraph,
            policy=pi,
            sGraph=sGraph,
            laoIter=laoIter,
            nonterminaltips=ntt,
            initial_value=initial_value,
            seed=seed
        )
