from msdm.core.algorithmclasses import Learns, Result
from msdm.core.problemclasses.tabularstochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from typing import Iterable
import numpy as np 

"""
Use prebuilt software for computing nash equilibria?(Gambit can do N>2 players, but requires separate installation)
"""
# Use nashpy for now 
class NashQ(Learns):
    
    def __init__(self,num_episodes=200):
        self.num_episodes = num_episodes
    
    def train_on(self,problem:TabularStochasticGame,agent_names: Iterable):
        pass 
    
    def update_agent(self,agent_name):
        pass 
    
    def nash_equilibrium(self):
        pass 