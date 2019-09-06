from pyrlap.domains.gridworld.gridworld import GridWorld
from pyrlap.domains.sharedcontrolgridworld import SharedControlGridWorld
''' 
A shared control gridworld that is partially observable


'''

class PO_SharedControlGridWorld(SharedControlGridWorld):

    def __init__(self):
        self.display = ''

    def update_display(self, state):
        self.display = state['mode']

    def return_display(self):
        return self.display


