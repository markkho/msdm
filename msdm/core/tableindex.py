from typing import Mapping, NamedTuple
import warnings
from itertools import product
from typing import Hashable, Sequence, Tuple, TypeVar, Union
from msdm.core.utils.funcutils import cached_property

FieldName = TypeVar("FieldName", bound=Hashable)
FieldValue = TypeVar("FieldValue", bound=Hashable)
FieldDomain = Sequence[FieldValue]

class domaintuple(tuple):
    def __new__(cls, elements):
        if isinstance(elements, cls):
            return elements
        return super().__new__(cls, elements)
    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"
    @cached_property
    def _index(self) -> Mapping[FieldValue,int]:
        return {e: ei for ei, e in enumerate(self)}
    def index(self, element) -> int:
        return self._index[element]
    def __hash__(self):
        return tuple.__hash__(self)

class Field(NamedTuple):
    name : FieldName
    domain : Union[domaintuple,Sequence[FieldValue]]
    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, domain={repr(self.domain)})"

class DomainError(BaseException): pass
class SliceError(ValueError): pass
class MultipleIndexError(IndexError): pass
class IndexSizeError(IndexError): pass

class TableIndex:
    """
    An ordered collection of named fields representing
    the indexing scheme of a table.
    """
    _FIELD_SLICE = slice(None)
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
            fields = [Field(n, domaintuple(v)) for n, v in zip(field_names, field_domains)]
        self._fields = tuple(fields)
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
    def product(self):
        yield from product(*self.field_domains)
    def product_dicts(self):
        for assignments in self.product():
            yield dict(zip(self.field_names, assignments))
    def __repr__(self):
        return f"{self.__class__.__name__}(fields={repr(self._fields)})"

    ########################################################
    #    Logic for indexing with arbitrary hashable keys   #
    ########################################################
    def _array_index(self, selector):
        # We first try to directly index into the outermost field
        try:
            idx = self.fields[0].domain.index(selector)
            return (idx,)
        except (KeyError, ValueError, TypeError):
            pass
        
        # Then we handle different selector types...
        
        # The simplest cases are if its just a slice or ellipsis
        if isinstance(selector, slice):
            if selector != self._FIELD_SLICE:
                raise SliceError("Only full field slices are allowed")
            return selector
        if selector == ...:
            return selector
        
        # Beyond this point, we only allow lists or tuples
        if not isinstance(selector, (list, tuple)):
            raise KeyError(f"Unable to resolve selector {selector} in {repr(self)}")
        if len(selector) == 1:
            if isinstance(selector[0], slice):
                if selector[0] != self._FIELD_SLICE:
                    raise SliceError("Only full field slices are allowed")
                return selector
            elif selector[0] == ...:
                return selector
            
        # If it is a list or domaintuple, we try to index into the outermost field
        if isinstance(selector, (list, domaintuple)):
            field_indices = self._index_into_domain(selector, self.fields[0].domain)
            field_indices = type(selector)(field_indices)
        
        # If it is a tuple, we try to index across fields 
        if isinstance(selector, tuple):
            field_indices = self._index_into_fields(selector)
            field_indices = tuple(field_indices)
        return field_indices

    def _updated_index(self, array_index):
        # short-circuiting if array_index is ellipses or all slices
        if (
            isinstance(array_index, slice) or \
            array_index == ... or \
            (
                len(array_index) == 1 and (
                    isinstance(array_index[0], slice) or \
                    array_index[0] == ...
                )
            ) or \
            all(isinstance(i, slice) or i == ... for i in array_index)
        ):
            return self
        
        # if its a list, then it refers to the outermost
        # field only.
        if isinstance(array_index, list):
            new_fields = [
                Field(
                    name=self.fields[0].name,
                    domain=domaintuple([self.fields[0].domain[i] for i in array_index])
                ),
                *self.fields[1:]
            ]
            return self.__class__(fields=new_fields)
        
        # otherwise, construct the new table index
        array_index = self._pad_out_ellipses(array_index)
        new_fields = []
        for fi, field in enumerate(self.fields):
            if fi > (len(array_index) - 1):
                new_fields.extend(self.fields[fi:])
                break
            field_index = array_index[fi]
            if field_index == slice(None):
                new_fields.append(field)
                continue
            # logic here assumes that array_index is only
            # a strict tuple if it is an element of the domain
            # so it depends on what self._array_index does
            if isinstance(field_index, (list, domaintuple)):
                new_domain = domaintuple([field.domain[i] for i in field_index])
                new_fields.append(Field(field.name, new_domain))
                continue
            
            # at this point, it must be a singleton so we don't include it in
            # the index - it will be dropped from the new index
        return self.__class__(fields=new_fields)
    def _index_into_fields(self, selector):
        selector = self._pad_out_ellipses(selector)
        if len(selector) > len(self):
            raise IndexSizeError(f"too many indices for index: index is {len(self)}-dimensional, but {len(selector)} were indexed")
        field_indices = []
        contains_sequence = False
        for field_selector, field in zip(selector, self.fields):
            # field_selectors are either singletons, subsets of a domain (lists or tuples), or slices
            if field_selector in field.domain:
                field_indices.append(field.domain.index(field_selector))
            elif field_selector == field.domain: # for short-circuiting
                if contains_sequence:
                    raise MultipleIndexError(
                        f"TableIndex does not support domain-value indexing on multiple fields like {selector}"
                    )
                contains_sequence = True
                field_indices.append(self._FIELD_SLICE)
            elif isinstance(field_selector, (list, tuple)):
                if contains_sequence:
                    raise MultipleIndexError(
                        f"TableIndex does not support domain-value indexing on multiple fields like {selector}"
                    )
                contains_sequence = True
                domain_indices = self._index_into_domain(field_selector, field.domain)
                field_indices.append(domain_indices)
            elif isinstance(field_selector, slice):
                if field_selector != self._FIELD_SLICE:
                    raise SliceError("Only full field slices are allowed")
                field_indices.append(field_selector)
            else:
                raise IndexError(f"Unrecognized field selector {field_selector}")
        return field_indices
    def _index_into_domain(self, field_selector, domain):
        try:
            return type(field_selector)([domain.index(e) for e in field_selector])
        except (ValueError, KeyError, TypeError):
            raise DomainError(f"Elements of {field_selector} not in {domain}")
    def _pad_out_ellipses(self, index):
        if ... in index:
            assert index.count(...) == 1
            ellipsis_idx = index.index(...)
            fields_left = len(self) - len(index) + 1
            index = \
                list(index[:ellipsis_idx]) + \
                [self._FIELD_SLICE]*fields_left + \
                list(index[ellipsis_idx + 1:])
        return index
