import inspect
import itertools
import functools
import numpy as np
import pandas as pd
from msdm.core.distributions import DictDistribution


class JointProbabilityTable(DictDistribution):
    """
    Representation of a distribution over variables.
    """
    @classmethod
    def null_table(cls):
        return null_joint_probability_table

    @classmethod
    def deterministic(cls, assignment):
        if isinstance(assignment, dict):
            assignment = Assignment.from_kwargs(**assignment)
        assert isinstance(assignment, Assignment)
        return JointProbabilityTable({assignment: 1.0})

    @classmethod
    def from_pairs(cls, pairs):
        """
        Construct a `JointProbabilityTable` from [assignment, prob] pairs.

        Parameters:
        ----------
        pairs : Iterable
            An iterable of [assignment, prob] pairs. `assignment`s can
            be dictionaries or `Assignment` objects.
        """
        table = {}
        for assignment, prob in pairs:
            if isinstance(assignment, dict):
                assignment = Assignment.from_kwargs(**assignment)
            assert isinstance(assignment, Assignment)
            table[assignment] = table.get(assignment, 0.0) + prob
        table = JointProbabilityTable(table)
        return table

    def first(self):
        return next(iter(self.items()))

    def variables(self):
        top_variables = list(self.first()[0].variables())
        return top_variables

    def normalize(self):
        normalizing_factor = sum([v for v in self.values()])
        table = {assignment: prob/normalizing_factor for assignment, prob in self.items()}
        return JointProbabilityTable(table)

    def _table_join(self, other):
        joint_table = {}
        for (self_assn, self_prob), (other_assn, other_prob) in itertools.product(self.items(), other.items()):
            if self_assn.compatible_with(other_assn):
                joint_prob = self_prob*other_prob
                if joint_prob > 0:
                    joint_table[self_assn.__quickadd__(other_assn)] = joint_prob
        joint_table = JointProbabilityTable(joint_table)
        return joint_table

    def _factor_join(self, factor):
        """
        """
        signature = factor.signature
        marg_join_dist = {}
        args = signature['input_variables']
        for assignment, prob in self.items():
            if prob == 0.:
                continue
            assignment_dict = dict(assignment)
            factor_result = factor(
                *[assignment_dict[arg] for arg in args]
            )
            if factor_result is None:
                marg_join_dist[assignment] = marg_join_dist.get(assignment, 0.) + prob
                continue

            # factor is a potential
            if isinstance(factor_result, (int, float)):
                if factor_result > 0.0:
                    marg_join_dist[assignment] = marg_join_dist.get(assignment, 0.) + prob*factor_result
                continue

            # factor is a conditional distribution
            for cond_assignment, cond_prob in factor_result.items():
                if cond_prob == 0.:
                    continue
                if not assignment.compatible_with(cond_assignment):
                    continue
                joint_prob = prob*cond_prob
                joint_assignment = cond_assignment.__quickadd__(assignment)
                marg_join_dist[joint_assignment] = \
                    marg_join_dist.get(joint_assignment, 0.0) + joint_prob
        return JointProbabilityTable(marg_join_dist)

    def _single_join(self, other):
        """Join two probability tables"""
        if isinstance(other, JointProbabilityTable):
            return self._table_join(other)
        elif hasattr(other, '_is_factor') and other._is_factor:
            return self._factor_join(other)
        else:
            raise ValueError("Unknown join type")

    def join(self, *others):
        table = self
        for other in others:
            table = table._single_join(other)
        return table

    def groupby(self, columns):
        """
        Parameters:
        ----------
        columns: float or callable from variables to True/False
        """
        marg = {}
        for assignment, prob in self.items():
            if isinstance(columns, (list, tuple)):
                marg_assignment = \
                    Assignment.from_pairs([(v, val) for v, val in assignment if v in columns])
            elif callable(columns):
                marg_assignment = \
                    Assignment.from_pairs([(v, val) for v, val in assignment if columns(v)])
            marg[marg_assignment] = marg.get(marg_assignment, 0.0) + prob
        return JointProbabilityTable(marg)

    def rename_columns(self, columns):
        assignments = list(self.keys())
        new_table = []
        for assignment in assignments:
            if isinstance(columns, dict):
                kwargs = {columns.get(v, v): val for v, val in assignment}
            elif callable(columns):
                kwargs = {columns(v): val for v, val in assignment}
            new_table.append([kwargs, self[assignment]])
        return JointProbabilityTable.from_pairs(new_table)

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
        if not np.isclose(sum(self.probs), 1.0):
            raise UnnormalizedDistributionError(f"Probabilities sum to {sum(self.probs)}")
        return True

    def __hash__(self):
        return hash((frozenset(self), frozenset(self.values())))

    def __ne__(self, other):
        return not (self == other)

    def _repr_html_(self):
        return self.as_dataframe().to_html()

    def as_dataframe(self):
        df = pd.DataFrame([{**a.to_dict(), "prob": prob} for a, prob in self.items()])
        return df

class ConflictingAssignmentsError(Exception):
    pass

class InconsistentVariablesError(Exception):
    pass

class UnnormalizedDistributionError(Exception):
    pass

class Assignment(frozenset):
    """
    Represents an assignment of values to variables.
    """
    @classmethod
    def from_pairs(cls, pairs):
        return Assignment(pairs)

    @classmethod
    def from_kwargs(cls, **kwargs):
        return Assignment(((k, v) for k, v in kwargs.items()))

    def variables(self):
        for k, v in self:
            yield k

    def to_dict(self):
        return {k: v for k, v in self}

    def items(self):
        for k, v in self:
            yield k, v

    def values(self):
        for k, v in self:
            yield v

    def __repr__(self):
        return f"Assignment.from_pairs(({', '.join([str(kv) for kv in self])}))"

    def __str__(self):
        return \
            "Assignment((\n" + \
            ",\n".join([f"  ({repr(k)}, {repr(v)})" for k, v in self]) + \
            "\n))"

    def __quickadd__(self, other):
        """Combine Assignment instances without checking for conflicts"""
        return Assignment(frozenset.__or__(self, other))

    def __add__(self, other):
        """We can combine assignments as long as the values of shared variables match"""
        combined, shorter = (dict(self), other) if len(self) >= len(other) else (dict(other), self)
        for k, v in shorter:
            if combined.get(k, v) != v:
                raise ConflictingKeyError(f"Values of '{k}' conflict: self={combined[k]}, other={v}")
            combined[k] = v
        return Assignment(combined.items())

    def shared_variables(self, other):
        return set(self.variables()) & set(other.variables())

    def compatible_with(self, other):
        shared_variables = self.shared_variables(other)
        if len(shared_variables) == 0:
            return True
        self_dict = dict(self)
        other_dict = dict(other)
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

null_joint_probability_table = JointProbabilityTable.from_pairs([
    [dict(), 1.0]
])
