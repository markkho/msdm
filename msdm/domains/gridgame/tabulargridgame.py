from functools import reduce
from itertools import combinations, cycle
import json, copy
import numpy as np
import matplotlib.pyplot as plt 
from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.stochasticgame import TabularStochasticGame
from msdm.core.distributions import DiscreteFactorTable as Pr

TERMINALSTATE = {"isTerminal": True}

class TabularGridGame(TabularStochasticGame):
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
                 collision_cost=0,
                 step_cost=-1,
                 fence_success_prob=.5,
                 collision_prob=None
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
        self.fence_success_prob = fence_success_prob
        self.collision_prob = collision_prob
        self.height = len(parsed)
        self.width = len(parsed[0])
        self.agents = agents 
        self.collision_prob = collision_prob
        super(TabularStochasticGame,self).__init__(agent_names=sorted([ag['name'] for ag in agents]))

        self._initState = initState

        #set up rewards
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.collision_cost = collision_cost

    def initial_state_dist(self):
        return Pr([self._initState,])
    
    def joint_actions(self,s):
        actions = [
            {'x': 0, 'y': 0},
            {'x': 1, 'y': 0},
            {'x': -1, 'y': 0},
            {'x': 0, 'y': 1},
            {'x': 0, 'y': -1},
        ]
        action_dict = {}
        for agentname in self.agent_names:
            action_dict[agentname] = (action for action in actions)
        return action_dict

    def is_absorbing(self, s):
        for an in self.agent_names:
            ag = s[an]
            for g in self.goals:
                if ((ag['x'], ag['y']) == (g['x'], g['y'])) and (an in g['owners']):
                    return True
        return False

    def is_terminal(self, s):
        return s.get('isTerminal', False)


    def next_state_dist(self, s, ja):
        if self.is_terminal(s):
            return Pr([TERMINALSTATE,])
        if self.is_absorbing(s):
            return Pr([TERMINALSTATE,])

        #agent-based transitions
        agentMoveDists = []
        for an in self.agent_names:
            agent = copy.deepcopy(s[an])

            # action effect
            EPS = .00001 # minor hack to handle agent collisions
            agentaction = ja[an]

            # assumes you can't leave the grid
            agent['x'] = max(min(agent['x'] + agentaction['x'], self.width-1), 0)
            agent['y'] = max(min(agent['y'] + agentaction['y'], self.height-1), 0)

            agentMove = Pr([{an:s[an]}, {an: agent}], probs=[EPS, 1-EPS])
            #fence-agent effects
            for fence in self.fences:
                if (self.same_location(s[an], fence['start'])) and self.same_location(agent, fence['end']):
                    fenceEffect = Pr([{an: s[an]}])
                    agentMove = agentMove * self.fence_success_prob | fenceEffect * (1 - self.fence_success_prob)

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
        collisions = []
        for ns in agentDist.support:
            ns = copy.deepcopy(ns)
            logit = 0
            for an0, an1 in combinations(self.agent_names, r=2):
                skip = False
                for goal in self.goals:
                    if an0 in goal["owners"] or an1 in goal["owners"]:
                        if self.same_location(goal,ns[an0]) or self.same_location(goal,ns[an1]):
                            skip = True
                if not skip:
                    # agents cant occupy the same location
                    if self.same_location(ns[an0], ns[an1]):
                        collisions.append((an0, an1))
                        logit += -np.inf
                    # agents can't swap locations
                    if self.same_location(ns[an0], s[an1]) and self.same_location(ns[an1], s[an0]):
                        logit += -np.inf
                    if self.collision_prob is not None and self.collision_prob != .5:
                        # NOTE: for this to properly handle collisions, it needs to split into two possible outcomes:
                        # one where each agent gets into the center tile
                        raise NotImplementedError("Currently does not handle probabilistic collisions!")
                # agents can't swap locations
                if self.same_location(ns[an0], s[an1]) and self.same_location(ns[an1], s[an0]):
                    logit += -np.inf
            interactions.append(ns)
            interactionLogits.append(logit)

        if self.collision_prob is not None and self.collision_prob != .5:
            # NOTE: for this to properly handle collisions, it needs to split into two possible outcomes:
            # one where each agent gets into the center tile
            raise NotImplementedError("Currently does not handle probabilistic collisions!")

        if self.collision_prob is None: #HACK - if there was any possibility of collision, nobody moves
            new_interaction_logits = []
            for ns, nslog in zip(interactions, interactionLogits):
                for an0, an1 in collisions:
                    if (ns[an0] != s[an0]) or (ns[an1] != s[an1]):
                        nslog = -np.inf
                new_interaction_logits.append(nslog)
            interactionLogits = new_interaction_logits

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
        if self.is_terminal(s) or self.is_terminal(ns):
            return jr
        for goal in self.goals:
            for agentName in goal['owners']:
                agent = ns[agentName]
                if self.same_location(goal, agent):
                    jr[agentName] += self.goal_reward

        for agentName, a in ja.items():
            if a != {'x': 0, 'y': 0}:
                jr[agentName] += self.step_cost

        # collision cost
        # HACK semantics are subtle - this simply looks at pairwise actions
        for name1, name2 in combinations(self.agent_names, 2):
            nloc1 = (s[name1]['x'] + ja[name1]['x'], s[name1]['y'] + ja[name1]['y'])
            nloc2 = (s[name2]['x'] + ja[name2]['x'], s[name2]['y'] + ja[name2]['y'])
            goal_state = False
            for goal in self.goals:
                if nloc1 == (goal["x"],goal["y"]) or nloc2 == (goal["x"],goal["y"]):
                    goal_state = True
            if nloc1 == nloc2 and not goal_state:
                jr[name1] += self.collision_cost
                jr[name2] += self.collision_cost

        return jr
    
    def plot(self,
             all_elements=False,
             figure=None,
             ax=None,
             figsize=None,
             figsize_multiplier=1,
             featurecolors=None,
             plot_walls=True,
             plot_initial_states=True,
             plot_fences=True,
             plot_absorbing_states=True
             ):
        if all_elements:
            plot_initial_states = True
            plot_absorbing_states = True
        from msdm.domains.gridgame.plotting import GridGamePlotter
        if featurecolors is None:
            
            featurecolors = {
                "fence": "brown",
                "obstacle": "black",
                "wall": "gray"
            }
        if ax is None:
            if figsize is None:
                figsize = (self.width * figsize_multiplier,
                           self.height * figsize_multiplier)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        gwp = GridGamePlotter(gg=self, ax=ax)
        gwp.plot_features(featurecolors=featurecolors)
        if plot_walls:
            gwp.plot_walls()
        if plot_initial_states:
            gwp.plot_initial_states(featurecolors=featurecolors)
        if plot_absorbing_states:
            gwp.plot_absorbing_states(featurecolors=featurecolors)
        if plot_fences:
            gwp.plot_fences()
        gwp.plot_outer_box()
        return gwp
      
    def animate(self,
             all_elements=False,
             figure=None,
             ax=None,
             figsize=None,
             figsize_multiplier=1,
             featurecolors=None,
             plot_walls=True,
             plot_initial_states=True,
             plot_fences=True,
             plot_absorbing_states=True
             ):
        
        if all_elements:
            plot_initial_states = True
            plot_absorbing_states = True
        from msdm.domains.gridgame.animating import GridGameAnimator
        
        if featurecolors is None:    
            featurecolors = {
                "fence": "brown",
                "obstacle": "black",
                "wall": "gray"
            }
            
                
        if ax is None:
            if figure is not None:
                raise Exception("Please pass in both figure and axis if frame is predefined")
            if figsize is None:
                figsize = (self.width * figsize_multiplier,
                           self.height * figsize_multiplier)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        gwp = GridGameAnimator(gg=self,figure=figure, ax=ax)
        gwp.plot_features(featurecolors=featurecolors)
        if plot_walls:
            gwp.plot_walls()
        if plot_initial_states:
            gwp.plot_initial_states()
        if plot_absorbing_states:
            gwp.plot_absorbing_states()
        if plot_fences:
            gwp.plot_fences()
        gwp.plot_outer_box()

        return gwp
        
    def full_initial_state(self):
        return {
            "agents": self._initState,
            "goals": self.goals,
            "obstacles": self.obstacles,
            "walls": self.walls,
            "fences": self.fences,
            "fence_success_prob": self.fence_success_prob,
            "height": self.height,
            "width": self.width
        }
