from typing import NamedTuple
from itertools import product
from typing import Hashable, Sequence, Tuple, TypeVar
from msdm.core.utils.funcutils import cached_property

FieldName = TypeVar("FieldName", bound=Hashable)
FieldValue = TypeVar("FieldValue", bound=Hashable)
FieldDomain = Sequence[FieldValue]

class Field(NamedTuple):
    name : FieldName
    domain : Sequence[FieldValue]
    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, domain={repr(self.domain)})"
    def domain_set(self):
        return frozenset(self.domain)
    def compatible_with(self, other : "Field") -> bool:
        """
        Two Field's with the same name are compatible 
        if their domains are the same up to permutation.
        """
        return (
            self.name == other.name and \
            self.domain_set() == other.domain_set()
        )
    def subsumes(self, other : "Field") -> bool:
        """
        `self` subsumes `other` if they have the same name
        and all of the elements of `other` are contained in
        `self`
        """
        return (
            self.name == other.name and \
            self.domain_set() >= other.domain_set()
        )
    def subsumed_by(self, other : "Field") -> bool:
        return other.subsumes(self)
    def permutation_of(self, other: "Field") -> Tuple[int]:
        """
        Returns indexes that would reorder the domain of
        `self` to be like that of `other`
        """
        self_domain_idx = {e: ei for ei, e in enumerate(self.domain)}
        domain_permutation = []
        for e in other.domain:
            domain_permutation.append(self_domain_idx[e])
        return tuple(domain_permutation)

class TableIndex:
    """
    An ordered collection of named fields representing
    the indexing scheme of a table.
    """
    def __init__(
        self,
        *,
        field_names : Sequence[FieldName] = None,
        field_domains : Sequence[FieldDomain] = None,
        fields : Sequence[Field] = None,
    ):
        if fields is None:
            if len(field_names) != len(field_domains):
                raise ValueError("Different numbers of fields names and domains")
            fields = tuple([Field(n, v) for n, v in zip(field_names, field_domains)])
        self._fields = fields
    def __getitem__(self, field_selection):
        if isinstance(field_selection, slice):
            return self.__class__(fields=self._fields[field_selection])
        return self._fields[field_selection]
    def __len__(self):
        return len(self._fields)
    @cached_property
    def field_names(self) -> Sequence[FieldName]:
        return tuple([v.name for v in self._fields])
    @cached_property
    def fields(self) -> Sequence[Field]:
        return tuple(self._fields)
    @cached_property
    def field_domains(self) -> Sequence[FieldDomain]:
        return tuple([v.domain for v in self._fields])
    @cached_property
    def shape(self) -> Tuple[int,...]:
        return tuple([len(v) for v in self.field_domains])
    def __eq__(self, other: "TableIndex") -> bool:
        """
        Two TableIndex's are *equal* if their fields and domains
        are ordered exactly the same.
        """
        return self._fields == other._fields
    def compatible_with(self, other: "TableIndex") -> bool:
        """
        Two TableIndex's are *compatible* if their field names and domains are
        equivalent up to permutation.
        """
        if set(self.field_names) != set(other.field_names):
            return False
        self_fields = set([(f.name, f.domain_set()) for f in self.fields])
        other_fields = set([(f.name, f.domain_set()) for f in other.fields])
        return self_fields == other_fields
    def subsumes(self, other: "TableIndex") -> bool:
        """
        A TableIndex A *subsumes* a TableIndex B if A and B have the same field
        names and the domain of every field in A is a superset of the corresponding
        domain in B.
        """
        if self == other or self.compatible_with(other):
            return True
        if set(self.field_names) != set(other.field_names):
            return False
        other_fields = {f.name: f for f in other.fields}
        for f in self.fields:
            if not f.subsumes(other_fields[f.name]):
                return False
        return True
    def subsumed_by(self, other: "TableIndex") -> bool:
        return other.subsumes(self)
    def reindexing_permutations(self, other : "TableIndex") -> Tuple[Tuple[int],Tuple[int]]:
        """
        If two TableIndex's are compatible, this returns how the field ordering
        and domain orderings of `self` can be permuted to match `other`.
        """
        assert self.subsumes(other), \
            f"Old index does not subsume the new one\nOld: {repr(self)}\nNew: {repr(other)}"
        field_permutation = [self.field_names.index(name) for name in other.field_names]
        other_name_fields = {f.name : f for f in other.fields}
        domain_permutations = []
        for self_field in self.fields:
            other_field = other_name_fields[self_field.name]
            domain_permutation = self_field.permutation_of(other_field)
            domain_permutations.append(domain_permutation)
        return tuple(field_permutation), tuple(domain_permutations)
    def product(self):
        yield from product(*self.field_domains)
    def product_dicts(self):
        for assignments in self.product():
            yield dict(zip(self.field_names, assignments))
    def __repr__(self):
        return f"{self.__class__.__name__}(fields={repr(self._fields)})"