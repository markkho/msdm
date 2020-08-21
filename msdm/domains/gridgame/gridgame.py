from functools import reduce
from itertools import combinations
import json, copy
import numpy as np

from msdm.core.utils.gridstringutils import stringToElementArray
from msdm.core.problemclasses.stochasticgame import StochasticGame
from msdm.core.distributions import DiscreteFactorTable as Pr

TERMINALSTATE = {"isTerminal": True}

class GridGame(StochasticGame):
    def __init__(self, 
                 gameString,
                 agentSymbols=("A0", "A1"),
                 goalSymbols=(
                     ("G0", ("A0", )), 
                     ("G1", ("A1", )), 
                     ("G", ("A0", "A1"))
                 ),
                 obstacleSymbols=("#",),
                 wallSymbols=(
                     ('[', 'left'),  
                     ( ']', 'right'), 
                     ( '^', 'above'), 
                     ( '_', 'below')
                 ),
                 fenceSymbols=(
                     ('{', 'left'),  
                     ( '}', 'right'), 
                     ( '~', 'above'), 
                     ( 'u', 'below')
                 ),
                 goalReward=10,
                 stepCost=-1,
                 fenceSuccessProb=.5
                ):
        # parse game string and convert to initial state game representation
        parseParams = {"colsep": None, "rowsep": "\n", "elementsep": "."}
        parsed = stringToElementArray(gameString, **parseParams)
        h = len(parsed)
        goalSymbols = dict(goalSymbols) if not isinstance(goalSymbols, dict) else goalSymbols
        wallSymbols = dict(wallSymbols) if not isinstance(wallSymbols, dict) else wallSymbols
        fenceSymbols = dict(fenceSymbols) if not isinstance(fenceSymbols, dict) else fenceSymbols
        
        goals, walls, fences, obstacles, agents = [], [], [], [], []
        for y_, row in enumerate(parsed):
            y = h - y_ - 1
            for x, cell in enumerate(row):
                for sym in cell:
                    if sym in goalSymbols:
                        owners = goalSymbols[sym]
                        goals.append({
                            'type': 'goal',
                            'x': x,
                            'y': y,
                            'owners': owners
                        })
                    elif sym in wallSymbols:
                        walls.append({
                            'type': 'wall',
                            'start': {
                                'x': x,
                                'y': y,    
                            },
                            'end': {
                                'x': x + {'left': -1, 'right': 1}.get(wallSymbols[sym], 0),
                                'y': y + {'below': -1, 'above': 1}.get(wallSymbols[sym], 0),
                            }
                        })
                    elif sym in fenceSymbols:
                        fences.append({
                            'type': 'fence',
                            'start': {
                                'x': x,
                                'y': y,    
                            },
                            'end': {
                                'x': x + {'left': -1, 'right': 1}.get(fenceSymbols[sym], 0),
                                'y': y + {'below': -1, 'above': 1}.get(fenceSymbols[sym], 0),
                            }
                        })
                    elif sym in obstacleSymbols:
                        obstacles.append({
                            'type': 'obstacle',
                            'x': x,
                            'y': y
                        })
                    elif sym in agentSymbols:
                        agents.append({
                            'type': 'agent',
                            'name': sym,
                            'x': x,
                            'y': y
                        })

        #current state representation only has agents as mutable part of state
        initState = {
            **{ag['name']: ag for ag in sorted(agents, key=lambda g: json.dumps(g, sort_keys=True))}
        }
        self.goals = sorted(goals, key=lambda g: json.dumps(g, sort_keys=True))
        self.obstacles = sorted(obstacles, key=lambda g: json.dumps(g, sort_keys=True))
        self.walls = sorted(walls, key=lambda g: json.dumps(g, sort_keys=True))
        self.fences = sorted(fences, key=lambda g: json.dumps(g, sort_keys=True))
        self.fenceSuccessProb = fenceSuccessProb
       
        super().__init__(agentNames=sorted([ag['name'] for ag in agents]))
        
        self._initState = initState
        
        #set up rewards
        self.goalReward = goalReward
        self.stepCost = stepCost
        
    def getInitialStateDist(self):
        return Pr([self._initState,])
   
    def getJointActionDist(self, s):
        actions = [
            {'x': 0, 'y': 0},
            {'x': 1, 'y': 0},
            {'x': -1, 'y': 0},
            {'x': 0, 'y': 1},
            {'x': 0, 'y': -1},
        ]
        adists = []
        for agentname in self.agentNames:
            adist = Pr([{agentname: action} for action in actions])
            adists.append(adist)
        return reduce(lambda a, b: a & b, adists)
    
    def isAbsorbing(self, s):
        for an in self.agentNames:
            ag = s[an]
            for g in self.goals:
                if ((ag['x'], ag['y']) == (g['x'], g['y'])) and (an in g['owners']):
                    return True
        return False

    def isTerminal(self, s):
        return s.get('isTerminal', False)

    def getNextStateDist(self, s, ja):
        if self.isAbsorbing(s):
            return Pr([TERMINALSTATE,])
        
        #agent-based transitions
        agentMoveDists = []
        for an in self.agentNames:
            agent = copy.deepcopy(s[an])
            
            # action effect
            EPS = 1e-5 #minor hack to handle agent collisions
            agentaction = ja[an]
            agent['x'] += agentaction['x']
            agent['y'] += agentaction['y']
            agentMove = Pr([{an:s[an]}, {an: agent}], probs=[EPS, 1-EPS])
            
            #fence-agent effects
            for fence in self.fences:
                if (self.sameLoc(s[an], fence['start'])) and self.sameLoc(agent, fence['end']):
                    fenceEffect = Pr([{an: s[an]}])
                    agentMove = agentMove*self.fenceSuccessProb | fenceEffect*(1 - self.fenceSuccessProb)
            
            #agent-obstacle interactions
            for obs in self.obstacles:
                if self.sameLoc(agent, obs):
                    obsConstraint = Pr([{an: s[an]}, {an: agent}], probs=[1, 0])
                    agentMove &= obsConstraint
                    
            #agent-wall interactions
            for wall in self.walls:
                if self.sameLoc(s[an], wall['start']) and self.sameLoc(agent, wall['end']):
                    wallConstraint = Pr([{an: s[an]}, {an: agent}], probs=[1, 0])
                    agentMove &= obsConstraint
                    
            agentMoveDists.append(agentMove)
            
        #compute joint distribution by combining independent distributions
        agentDist = reduce(lambda a, b: a & b, agentMoveDists)
        
        #calculate agent interactions
        interactions = []
        interactionLogits = []
        for ns in agentDist.support:
            ns = copy.deepcopy(ns)
            logit = 0
            for an0, an1 in combinations(self.agentNames, r=2):
                if self.sameLoc(ns[an0], ns[an1]):
                    logit += -np.inf
            interactions.append(ns)
            interactionLogits.append(logit)
        interactionEffects = Pr(interactions, logits=interactionLogits)
        return agentDist & interactionEffects
   
    def inGoal(self, s, agentName):
        goals = {n: o for n, o in s.items() if 'goal' in n}
        agent = s[agentName]
        for g in goals.values():
            if ((agent['x'], agent['y']) == (g['x'], g['y'])) and (agent['name'] in g['owners']):
                return True
        return False
    
    def sameLoc(self, a, b):
        return (a['x'], a['y']) == (b['x'], b['y'])

    def getJointRewards(self, s, ja, ns):
        jr = {an: 0 for an in self.agentNames}
        
        for goal in self.goals:
            for agentName in goal['owners']:
                agent = ns[agentName]
                if self.sameLoc(goal, agent):
                    jr[agentName] += self.goalReward
                    
        for agentName, a in ja.items():
            if a != {'x': 0, 'y': 0}:
                jr[agentName] += self.stepCost
        return jr

if __name__ == "__main__":
    #example usage
    gameString = """
        #  # # #  G0 #  # # # 
        G0 . . A0 .  A1 . . G1
        #  # # #  G1 #  # # #
    """.strip()

    gg = GridGame(gameString)
    s = gg.getInitialStateDist().sample()
    a = {'A0': {'x': 1, 'y': 0}, 'A1': {'x': -1, 'y': 0}}
    nsdist = gg.getNextStateDist(s, a)
    ns = nsdist.sample()
    r = gg.getJointRewards(s, a, ns)
