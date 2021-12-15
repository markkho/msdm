from collections import defaultdict
import itertools
import numpy as np
from msdm.core.distributions import DictDistribution

class JointProbabilityTable(DictDistribution):
    """
    Representation of a distribution over variables.
    `implicit_prob` is set to 0 by default,
    which means un-represented rows can be ignored.
    If `implicit_prob` is set to a non-zero value,
    un-represented rows that are created when joining
    with other JointProbabilityTable instances will
    assume the input tables assign that value to
    that variable assignment.
    """
    implicit_prob = 0

    @classmethod
    def from_pairs(cls, pairs, implicit_prob=0):
        """
        Construct a `JointProbabilityTable` from [assignment, prob] pairs.

        Parameters:
        ----------
        pairs : Iterable
            An iterable of [assignment, prob] pairs. `assignment`s can
            be dictionaries or `Assignment` objects.
        implicit_prob : float
        """
        table = defaultdict(float)
        for assignment, prob in pairs:
            if isinstance(assignment, dict):
                assignment = Assignment.from_kwargs(**assignment)
            assert isinstance(assignment, Assignment)
            table[assignment] += prob
        table = JointProbabilityTable(table)
        table.implicit_prob = implicit_prob
        return table

    def variables(self):
        top_variables = list(next(iter(self.keys())).variables())
        return top_variables

    def normalize(self):
        assert self.implicit_prob == 0., f"Cannot normalize if implicit_prob value is {self.implicit_prob}"
        normalizing_factor = sum([v for v in self.values()])
        table = {assignment: prob/normalizing_factor for assignment, prob in self.items()}
        return JointProbabilityTable(table)

    def join(self, other):
        """Join two probability tables"""
        # first pass - find compatible assignments and combine values, tracking which are
        # unique to each table
        self_outer = dict(self.items())
        other_outer = dict(other.items())
        joint_table = {}
        for (self_assn, self_prob), (other_assn, other_prob) in itertools.product(self.items(), other.items()):
            if self_assn.compatible_with(other_assn):
                joint_prob = self_prob*other_prob
                if joint_prob > 0:
                    joint_table[self_assn + other_assn] = joint_prob
                if self_assn in self_outer:
                    del self_outer[self_assn]
                if other_assn in other_outer:
                    del other_outer[other_assn]

        # second pass - if any appear only in the outer of either table, use non-zero implicit_prob values
        assert other.implicit_prob >= 0
        if other.implicit_prob > 0:
            for self_assn, self_prob in self_outer.items():
                joint_table[self_assn] = self_prob*other.implicit_prob

        assert self.implicit_prob >= 0
        if self.implicit_prob > 0:
            for other_assn, other_prob in other_outer.items():
                joint_table[other_assn] = other_prob*self.implicit_prob
        joint_table = JointProbabilityTable(joint_table)
        joint_table.implicit_prob = other.implicit_prob*self.implicit_prob
        return joint_table

    def _check_valid(self):
        # check that all assignments are over the same set of variables
        top_variables = set(next(iter(self.keys())).variables())
        for assignment in self.keys():
            variables = set(assignment.variables())
            if top_variables != variables:
                raise InconsistentVariablesError(
                    f"Table contains rows with different variables: {top_variables} != {variables}"
                )

        #check that probabilities sum to 1
        if self.implicit_prob != 0:
            raise UnnormalizedDistributionError(
                f"Table is only normalized if implicit_prob == 0 (self.implicit_prob = {self.implicit_prob})"
            )
        if self.implicit_prob == 0 and not np.isclose(sum(self.probs), 1.0):
            raise UnnormalizedDistributionError(f"Probabilities sum to {sum(self.probs)}")

    def __eq__(self, other):
        print(self.implicit_prob, other.implicit_prob)
        if self.implicit_prob != other.implicit_prob:
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not (self == other)

class InconsistentVariablesError(Exception):
    pass
class UnnormalizedDistributionError(Exception):
    pass

class Assignment(frozenset):
    """
    Represents an assignment of values to variables.
    """
    @classmethod
    def from_kwargs(cls, **kwargs):
        return Assignment(((k, v) for k, v in kwargs.items()))

    def variables(self):
        for k, v in self:
            yield k

    def __repr__(self):
        return f"Assignment(({', '.join([str(kv) for kv in self])}))"

    def __str__(self):
        return \
            "Assignment((\n" + \
            ",\n".join([f"  ('{k}', {v})" for k, v in self]) + \
            "\n))"

    def __add__(self, other):
        """We can combine assignments as long as the values of shared variables match"""
        combined = {}
        for k, v in itertools.chain(self, other):
            if combined.get(k, v) != v:
                raise ConflictingKeyError(f"Values of '{k}' conflict: self={combined[k]}, other={v}")
            combined[k] = v
        return Assignment.from_kwargs(**combined)

    def to_dict(self):
        return {k: v for k, v in self}

    def items(self):
        for k, v in self:
            yield k, v

    def shared_variables(self, other):
        return set(self.variables()) & set(other.variables())

    def compatible_with(self, other):
        shared_variables = self.shared_variables(other)
        if len(shared_variables) == 0:
            return True
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        for k in shared_variables:
            if self_dict[k] != other_dict[k]:
                return False
        return True

    def __getitem__(self, variable):
        for k, v in self:
            if k == variable:
                return v

class ConflictingKeyError(Exception):
    pass
