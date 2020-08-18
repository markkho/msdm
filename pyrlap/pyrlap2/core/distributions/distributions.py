from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass
