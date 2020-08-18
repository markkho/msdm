import json
from itertools import chain

class AssignmentSet:
    def __init__(self, items=()):
        self._items = set([])
        for i in items:
            self._items.add(self.encode_item(i))

    def encode_item(self, i):
        if isinstance(i, dict):
            i = json.dumps(i, sort_keys=True)
        return i

    def decode_item(self, i):
        try:
            i = json.loads(i)
        except json.JSONDecodeError:
            pass
        except TypeError:
            pass
        return i

    def add(self, i):
        self._items.add(self.encode_item(i))
    
    def remove(self, i):
        self._items.remove(self.encode_item(i))
    
    def __and__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items & other._items)
    
    def __or__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items | other._items)
    
    def __sub__(self, other: "AssignmentSet"):
        return AssignmentSet(self._items - other._items)

    def __contains__(self, i):
        return self._items.__contains__(self.encode_item(i))
    
    def __iter__(self):
        for i in self._items:
            yield self.decode_item(i)

    def __len__(self):
        return len(self._items)

    def pop(self):
        return self.decode_item(self._items.pop())
    
    def __repr__(self):
        return self._items.__repr__()
    
    def __str__(self):
        return self._items.__str__()
