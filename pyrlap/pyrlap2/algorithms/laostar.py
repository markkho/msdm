from types import SimpleNamespace
import warnings, tqdm
import json
import random
import numpy as np

from pyrlap.pyrlap2.core.assignmentmap import AssignmentMap as Dict

def _hash(x):
    if isinstance(x, dict):
        return json.dumps(x, sort_keys=True)
    return x

class LAOStar:
    """
    Hansen, E. A., & Zilberstein, S. (2001). LAO*: 
    A heuristic search algorithm that finds solutions 
    with loops. Artificial Intelligence, 129(1-2), 35-62.
    """
    def __init__(self):
        pass

    def planOn(self, 
               mdp, 
               heuristic: "Function over states", 
               eGraph=None,
               showWarning=False,
               showProgress=True,
               maxLAOIters=100,
               policyEvaluationIters=100,
               policyIterationIters=100,
               seed=None
              ):
        if seed is None:
            seed = random.randint(1, 1e20)
        random.seed(seed)
        
        discountRate = 1 - mdp.terminationProb
        
        #initialize explicit graph
        if eGraph is None:
            eGraph = {} #explicit graph
        initStates = mdp.getInitialStateDist().support
        for s0 in initStates:
            actionorder = list(mdp.getActionDist(s0).support)
            random.shuffle(actionorder)
            node = {
                "parents": [], 
                "actionchildren": Dict(),
                "state": s0, 
                "value": heuristic(s0),
                "bestaction": actionorder[0],
                "actionorder": actionorder,
                "visitorder": len(eGraph),
                "expandedorder": -1,
                "expanded": False
            }
            eGraph[_hash(s0)] = node

        def policyImprovement(graph, eGraph):
            pichange = False
            for n in graph.values():
                s = n["state"]
                aa = n["actionorder"]
                avals = []
                maxa = n["bestaction"]
                maxav = -np.inf
                for a in aa:
                    aval = 0
                    for ns, p in mdp.getNextStateDist(s, a).items(probs=True):
                        aval += p*(mdp.getReward(s, a, ns)+discountRate*eGraph[_hash(ns)]["value"])
                    if aval > maxav:
                        maxav = aval
                        maxa = a
                if n["bestaction"] != maxa:
                    pichange = True
                n["bestaction"] = maxa
            return pichange

        def policyEvaluation(graph, eGraph):
            #NOTE: this can fail to converge if the current policy is stuck in a loop
            #TODO: do policy evaluation with matrix inversion / linear equation solving
            for iPE in range(policyEvaluationIters):
                endPE = False
                valchange = -np.inf
                for n in graph.values():
                    assert n['expanded']
                    s, a = n["state"], n["bestaction"]
                    nsdist = mdp.getNextStateDist(s, a).items(probs=True)
                    expval = 0
                    for ns, p in nsdist:
                        nextnode = eGraph[_hash(ns)]
                        expval += p*(mdp.getReward(s, a, ns) + discountRate*nextnode["value"])
                    valchange = np.max([np.abs(n["value"] - expval), valchange])
                    n["value"] = expval
                if valchange < 1e-6:
                    break
                if iPE == (policyEvaluationIters - 1) and showWarning:
                    #Note: whenever backtracking occurs, this will fail to converge
                    warnings.warn(f"Policy evaluation did not converge after {policyEvaluationIters} iterations")

        def updateDP(graph, eGraph):
            #run policy iteration on all the states in the subgraph graph
            # it is assumed that all of the nodes in graph are expanded
            policyImprovement(graph, eGraph)
            for iDP in range(policyIterationIters):
                policyEvaluation(graph, eGraph)
                pichange = policyImprovement(graph, eGraph)
                if not pichange:
                    break

        def getNonterminalTips(sGraph):
            ntt = []
            for n in sGraph.values():
                if (not n['expanded']) and (not mdp.isTerminal(n['state'])):
                    ntt.append(n)
            return ntt

        def getSolutionGraph(eGraph, initStates):
            sGraph = {}
            for s0 in initStates:
                toget = [s0, ]
                while len(toget) > 0:
                    s = toget.pop()
                    if _hash(s) in sGraph:
                        continue
                    n = eGraph[_hash(s)]
                    sGraph[_hash(s)] = n            
                    bestchildren = n['actionchildren'].get(n['bestaction'], [])
                    bestchildrenstates = [cn['state'] for cn in bestchildren]
                    toget.extend(bestchildrenstates)
            return sGraph

        def expandGraph(eGraph, n, nExpanded):
            s = n['state']
            n['expanded'] = True
            n['expandedorder'] = nExpanded
            aa = n['actionorder']
            for a in aa:
                children = []
                nextstates = list(mdp.getNextStateDist(s, a).support)
                random.shuffle(nextstates)
                for ns in nextstates:
                    if _hash(ns) not in eGraph:
                        actionorder = list(mdp.getActionDist(ns).support)
                        random.shuffle(actionorder)
                        nextnode = {
                            "state": ns,
                            "value": heuristic(ns),
                            "bestaction": actionorder[0],
                            "actionorder": actionorder,
                            "visitorder": len(eGraph),
                            "expandedorder": -1,
                            "parents": [n, ],
                            "actionchildren": Dict(),
                            "expanded": False
                        }
                        eGraph[_hash(ns)] = nextnode
                    else:
                        nextnode = eGraph[_hash(ns)]
                        nextnode['parents'].append(n)
                    children.append(nextnode)
                n['actionchildren'][_hash(a)] = children

        def getAncestors(eGraph, tip):
            ans = {}
            toget = [tip,]
            while len(toget) > 0:
                n = toget.pop()
                hs = _hash(n['state'])
                ans[hs] = n
                for pn in n['parents']:
                    if _hash(pn["state"]) in ans:
                        continue
                    bestchildren = pn['actionchildren'][pn['bestaction']]
                    if _hash(n['state']) in [_hash(cn['state']) for cn in bestchildren]:
                        toget.append(pn)
            return ans


        if showProgress:
            pbar = tqdm.tqdm()
        nExpanded = 0
        for s0 in initStates:
            n0 = eGraph[_hash(s0)]
            expandGraph(eGraph, n0, nExpanded)
            nExpanded += 1
            z = getAncestors(eGraph, n0)
            updateDP(z, eGraph)
        sGraph = getSolutionGraph(eGraph, initStates)
        for laoIter in range(maxLAOIters):
            if showProgress:
                pbar.update(1)
                pbar.set_description(f"|eGraph|: {len(eGraph)}; |sGraph| = {len(sGraph)}")
            
            ntt = getNonterminalTips(sGraph)
            if len(ntt) == 0:
                break
            nonterm = max(ntt, key=lambda n: (n["value"], -n["visitorder"]))
            expandGraph(eGraph, nonterm, nExpanded)
            nExpanded += 1
            z = getAncestors(eGraph, nonterm)
            updateDP(z, eGraph)
            sGraph = getSolutionGraph(eGraph, initStates)
            
        if showProgress:
            pbar.close()
            
        return SimpleNamespace(
            eGraph=eGraph,
            sGraph=sGraph,
            laoIter=laoIter,
            nonterminaltips=ntt,
            seed=seed
        )