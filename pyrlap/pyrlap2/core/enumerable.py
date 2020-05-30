from abc import ABC, abstractmethod
from typing import Iterable

class Enumerable(ABC):
    @abstractmethod
    def items(self) -> Iterable:
        pass

    @abstractmethod
    def keys(self) -> Iterable:
        pass

    @abstractmethod
    def asMatrix(self):
        pass

    def asDict(self):
        return dict(self.items())
