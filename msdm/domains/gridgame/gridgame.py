from functools import reduce
from itertools import combinations
import json, copy
import numpy as np

from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.problemclasses.stochasticgame import StochasticGame
from msdm.core.distributions import DiscreteFactorTable as Pr

TERMINALSTATE = {"isTerminal": True}

class GridGame(StochasticGame):
    def __init__(self,
                 game_string,
                 agent_symbols=("A0", "A1"),
                 goal_symbols=(
                     ("G0", ("A0", )), 
                     ("G1", ("A1", )), 
                     ("G", ("A0", "A1"))
                 ),
                 obstacle_symbols=("#",),
                 wall_symbols=(
                     ('[', 'left'),  
                     ( ']', 'right'), 
                     ( '^', 'above'), 
                     ( '_', 'below')
                 ),
                 fence_symbols=(
                     ('{', 'left'),  
                     ( '}', 'right'), 
                     ( '~', 'above'), 
                     ( 'u', 'below')
                 ),
                 goal_reward=10,
                 step_cost=-1,
                 fence_success_prob=.5
                 ):
        # parse game string and convert to initial state game representation
        parseParams = {"colsep": None, "rowsep": "\n", "elementsep": "."}
        parsed = string_to_element_array(game_string, **parseParams)
        h = len(parsed)
        goal_symbols = dict(goal_symbols) if not isinstance(goal_symbols, dict) else goal_symbols
        wall_symbols = dict(wall_symbols) if not isinstance(wall_symbols, dict) else wall_symbols
        fence_symbols = dict(fence_symbols) if not isinstance(fence_symbols, dict) else fence_symbols
        
        goals, walls, fences, obstacles, agents = [], [], [], [], []
        for y_, row in enumerate(parsed):
            y = h - y_ - 1
            for x, cell in enumerate(row):
                for sym in cell:
                    if sym in goal_symbols:
                        owners = goal_symbols[sym]
                        goals.append({
                            'type': 'goal',
                            'x': x,
                            'y': y,
                            'owners': owners
                        })
                    elif sym in wall_symbols:
                        walls.append({
                            'type': 'wall',
                            'start': {
                                'x': x,
                                'y': y,    
                            },
                            'end': {
                                'x': x + {'left': -1, 'right': 1}.get(wall_symbols[sym], 0),
                                'y': y + {'below': -1, 'above': 1}.get(wall_symbols[sym], 0),
                            }
                        })
                    elif sym in fence_symbols:
                        fences.append({
                            'type': 'fence',
                            'start': {
                                'x': x,
                                'y': y,    
                            },
                            'end': {
                                'x': x + {'left': -1, 'right': 1}.get(fence_symbols[sym], 0),
                                'y': y + {'below': -1, 'above': 1}.get(fence_symbols[sym], 0),
                            }
                        })
                    elif sym in obstacle_symbols:
                        obstacles.append({
                            'type': 'obstacle',
                            'x': x,
                            'y': y
                        })
                    elif sym in agent_symbols:
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
        self.fenceSuccessProb = fence_success_prob
       
        super().__init__(agent_names=sorted([ag['name'] for ag in agents]))
        
        self._initState = initState
        
        #set up rewards
        self.goalReward = goal_reward
        self.stepCost = step_cost
        
    def initial_state_dist(self):
        return Pr([self._initState,])
   
    def joint_action_dist(self, s):
        actions = [
            {'x': 0, 'y': 0},
            {'x': 1, 'y': 0},
            {'x': -1, 'y': 0},
            {'x': 0, 'y': 1},
            {'x': 0, 'y': -1},
        ]
        adists = []
        for agentname in self.agent_names:
            adist = Pr([{agentname: action} for action in actions])
            adists.append(adist)
        return reduce(lambda a, b: a & b, adists)
    
    def isAbsorbing(self, s):
        for an in self.agent_names:
            ag = s[an]
            for g in self.goals:
                if ((ag['x'], ag['y']) == (g['x'], g['y'])) and (an in g['owners']):
                    return True
        return False

    def is_terminal(self, s):
        return s.get('isTerminal', False)

    def next_state_dist(self, s, ja):
        if self.isAbsorbing(s):
            return Pr([TERMINALSTATE,])
        
        #agent-based transitions
        agentMoveDists = []
        for an in self.agent_names:
            agent = copy.deepcopy(s[an])
            
            # action effect
            EPS = 1e-5 # minor hack to handle agent collisions
            agentaction = ja[an]
            agent['x'] += agentaction['x']
            agent['y'] += agentaction['y']
            agentMove = Pr([{an:s[an]}, {an: agent}], probs=[EPS, 1-EPS])
            
            #fence-agent effects
            for fence in self.fences:
                if (self.same_location(s[an], fence['start'])) and self.same_location(agent, fence['end']):
                    fenceEffect = Pr([{an: s[an]}])
                    agentMove = agentMove*self.fenceSuccessProb | fenceEffect*(1 - self.fenceSuccessProb)
            
            #agent-obstacle interactions
            for obs in self.obstacles:
                if self.same_location(agent, obs):
                    obsConstraint = Pr([{an: s[an]}, {an: agent}], probs=[1, 0])
                    agentMove &= obsConstraint
                    
            #agent-wall interactions
            for wall in self.walls:
                if self.same_location(s[an], wall['start']) and self.same_location(agent, wall['end']):
                    wallConstraint = Pr([{an: s[an]}, {an: agent}], probs=[1, 0])
                    agentMove &= wallConstraint
                    
            agentMoveDists.append(agentMove)
            
        #compute joint distribution by combining independent distributions
        agentDist = reduce(lambda a, b: a & b, agentMoveDists)
        
        #calculate agent interactions
        interactions = []
        interactionLogits = []
        for ns in agentDist.support:
            ns = copy.deepcopy(ns)
            logit = 0
            for an0, an1 in combinations(self.agent_names, r=2):
                if self.same_location(ns[an0], ns[an1]):
                    logit += -np.inf
            interactions.append(ns)
            interactionLogits.append(logit)
        interactionEffects = Pr(interactions, logits=interactionLogits)
        return agentDist & interactionEffects
   
    def in_goal(self, s, agentname):
        goals = {n: o for n, o in s.items() if 'goal' in n}
        agent = s[agentname]
        for g in goals.values():
            if ((agent['x'], agent['y']) == (g['x'], g['y'])) and (agent['name'] in g['owners']):
                return True
        return False
    
    def same_location(self, a, b):
        return (a['x'], a['y']) == (b['x'], b['y'])

    def joint_rewards(self, s, ja, ns):
        jr = {an: 0 for an in self.agent_names}
        
        for goal in self.goals:
            for agentName in goal['owners']:
                agent = ns[agentName]
                if self.same_location(goal, agent):
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
    s = gg.initial_state_dist().sample()
    a = {'A0': {'x': 1, 'y': 0}, 'A1': {'x': -1, 'y': 0}}
    nsdist = gg.next_state_dist(s, a)
    ns = nsdist.sample()
    r = gg.joint_rewards(s, a, ns)
