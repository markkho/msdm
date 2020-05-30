from abc import ABC, abstractmethod
from pyrlap.pyrlap2.core.mdp.mdp import MarkovDecisionProcess

class Plottable(ABC):
    @abstractmethod
    def plot(self, ax=None, figsize=None, figsizeMult=None, **kwargs):
        pass

class Plotter(ABC):
    def __init__(self, mdp: MarkovDecisionProcess, ax=None):
        self.mdp = mdp
        self.ax = ax

    def title(self, title: str, **kwargs):
        self.ax.set_title(title, **kwargs)
        return self

    @abstractmethod
    def plot(self, state=None) -> "Plotter":
        pass
