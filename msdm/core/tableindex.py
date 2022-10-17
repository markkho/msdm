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
            fields = [Field(n, v) for n, v in zip(field_names, field_domains)]
        self._fields = fields
    def __getitem__(self, field_selection):
        if isinstance(field_selection, slice):
            return self.__class__(fields=self._fields[field_selection])
        return self._fields[field_selection]
    def __len__(self):
        return len(self._fields)
    @cached_property
    def field_names(self):
        return tuple([v.name for v in self._fields])
    @cached_property
    def fields(self):
        return tuple(self._fields)
    @cached_property
    def field_domains(self):
        return tuple([v.domain for v in self._fields])
    @cached_property
    def shape(self):
        return tuple([len(v) for v in self.field_domains])
    def domain_of(self, name: FieldName):
        return self[self.field_names.index(name)].domain
    def compatible_with(self, other: "TableIndex") -> bool:
        """
        Two TableIndex's are compatible if their field names and domains are
        equivalent up to permutation.
        """
        if set(self.field_names) != set(other.field_names):
            return False
        self_fields = set([(f.name, frozenset(f.domain)) for f in self.fields])
        other_fields = set([(f.name, frozenset(f.domain)) for f in other.fields])
        return self_fields == other_fields
    def reindexing_permutations(self, other : "TableIndex") -> Tuple[Tuple[int],Tuple[int]]:
        """
        If two TableIndex's are compatible, this returns how the field ordering
        and domain orderings of `self` can be permuted to match `other`.
        """
        assert self.compatible_with(other), \
            f"Index not compatible\nOld: {repr(self)}\nNew: {repr(other)}"
        field_permutation = [self.field_names.index(name) for name in other.field_names]
        domain_permutations = []
        for name, self_domain in self.fields:
            self_domain_idx = {e: ei for ei, e in enumerate(self_domain)}
            domain_permutation = tuple([self_domain_idx[e] for e in other.domain_of(name)])
            domain_permutations.append(domain_permutation)
        return tuple(field_permutation), tuple(domain_permutations)
    def product(self):
        yield from product(*self.field_domains)
    def product_dicts(self):
        for assignments in self.product():
            yield dict(zip(self.field_names, assignments))
    def equivalent_to(self, other : "TableIndex"):
        return self._fields == other._fields
    def __repr__(self):
        return f"{self.__class__.__name__}(fields={repr(self._fields)})"