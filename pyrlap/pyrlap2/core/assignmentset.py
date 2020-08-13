import json
from itertools import chain

class AssignmentSet:
    def __init__(self, items=()):
        self._items = set([])
        for i in items:
            if isinstance(i, dict):
                i = json.dumps(i, sort_keys=True)
            self._items.add(i)
    
    def add(self, i):
        if isinstance(i, dict):
            i = json.dumps(i, sort_keys=True)
        self._items.add(i)
    
    def remove(self, i):
        if isinstance(i, dict):
            i = json.dumps(i, sort_keys=True)    
        self._items.remove(i)
    
    def __and__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items & other._items)
    
    def __or__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items | other._items)
    
    def __sub__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items - other._items)

    def __contains__(self, i):
        if isinstance(i, dict):
            i = json.dumps(i, sort_keys=True)     
        return self._items.__contains__(i)
    
    def __iter__(self):
        for i in self._items:
            try:
                i = json.loads(i)
            except json.JSONDecodeError:
                pass
            yield i

    def __len__(self):
        return len(self._items)

    def pop(self):
        i = self._items.pop()
        try:
            i = json.loads(i)
        except json.JSONDecodeError:
            pass
        return i
    
    def __repr__(self):
        return self._items.__repr__()
    
    def __str__(self):
        return self._items.__str__()