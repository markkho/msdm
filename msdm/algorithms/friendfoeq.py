from msdm.core.algorithmclasses import Learns, Result
from msdm.core.problemclasses.stochasticgame import StochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from typing import Iterable
import numpy as np 

class FriendFoeQ(Learns):
    
    def __init__(self,num_episodes=200):
        self.num_episodes = num_episodes 
    
    def train_on(self,problem:StochasticGame,agent_names: Iterable, friends: dict, foes: dict) -> Result:
        pass 
    
    