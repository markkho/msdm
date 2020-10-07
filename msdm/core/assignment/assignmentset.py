import json
from itertools import chain
# from .assignmentmap import encode_item, decode_item

class AssignmentSet:
    def __init__(self, items=()):
        self._encoded_keys = {} #note: this isn't doing garbage collection
        self._items = set([])
        for i in items:
            self._items.add(self.encode_item(i))


    def encode_item(self, i):
        if isinstance(i, (dict, list)):
            encoded = json.dumps(i, sort_keys=True)
            self._encoded_keys[encoded] = i
            i = encoded
        return i

    def decode_item(self, encoded_item):
        try:
            return self._encoded_keys[encoded_item]
        except KeyError:
            return encoded_item

    def add(self, i):
        self._items.add(self.encode_item(i))
    
    def remove(self, i):
        self._items.remove(self.encode_item(i))

    def __merge__(self, other, new_set):
        merged = AssignmentSet()
        for ei in new_set:
            try:
                i = self.decode_item(ei)
            except KeyError:
                i = other.decode_item(ei)
            merged.add(i)
        return merged

    def __and__(self, other: "AssignmentSet"):
        return self.__merge__(other, self._items & other._items)
        # return AssignmentSet(self._items & other._items)
    
    def __or__(self, other: "AssignmentSet"):
        return self.__merge__(other, self._items | other._items)

    def __sub__(self, other: "AssignmentSet"):
        return self.__merge__(other, self._items - other._items)
        # return AssignmentSet(self._items - other._items)

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
