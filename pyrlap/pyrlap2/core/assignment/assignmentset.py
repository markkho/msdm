import json
from itertools import chain
from .assignmentmap import encode_item, decode_item

class AssignmentSet:
    def __init__(self, items=()):
        self._items = set([])
        for i in items:
            self._items.add(encode_item(i))

    def add(self, i):
        self._items.add(encode_item(i))
    
    def remove(self, i):
        self._items.remove(encode_item(i))
    
    def __and__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items & other._items)
    
    def __or__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items | other._items)
    
    def __sub__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items - other._items)

    def __contains__(self, i):
        return self._items.__contains__(encode_item(i))
    
    def __iter__(self):
        for i in self._items:
            yield decode_item(i)

    def __len__(self):
        return len(self._items)

    def pop(self):
        return decode_item(self._items.pop())
    
    def __repr__(self):
        return self._items.__repr__()
    
    def __str__(self):
        return self._items.__str__()
