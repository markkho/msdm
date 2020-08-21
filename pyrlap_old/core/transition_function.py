from collections import defaultdict, Mapping
from typing import Hashable

class TransitionFunction(Mapping):
    def __init__(self, tf_dict):
        self.tf_dict = tf_dict
        self.inv_dict = None

    def __getitem__(self, key):
        return self.tf_dict[self.__keytransform__(key)]

    def __delitem__(self, key):
        del self.tf_dict[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.tf_dict)

    def __len__(self):
        return len(self.tf_dict)

    def __keytransform__(self, key):
        return key

    def _construct_inv(self):
        inv_dict = defaultdict(set)
        for s in self:
            for a in self[s]:
                for ns in self[s][a]:
                    inv_dict[(ns, a)].add(s)
        self.inv_dict = inv_dict

    def inv(self, ns : Hashable, a : Hashable = None):
        """
        Returns the pre-image of a state in the transition function. E.g.
        tf.inv(s, a) = {s_ s.t. tf[s][a][s_] > 0}
        tf.inv(s) = {s_ s.t. ∃a s.t. tf[s][a][s_] > 0}
        """
        if a is None:
            all_preimg = set([])
            all_preimg.update(*[self.inv(ns, a) for a in self.tf_dict[ns]])
            return all_preimg

        if self.inv_dict is None:
            self._construct_inv()

        return self.inv_dict[(ns, a)]

