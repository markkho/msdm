import numpy as np
from typing import Mapping, Sequence
from msdm.core.problemclasses.mdp.mdp import Action, State
from msdm.core.table import Table, TableIndex

Vector = Sequence[float]
Matrix = Sequence[Vector]
StateList = Sequence[State]
ActionList = Sequence[Action]

class StateTable(Table):
    @classmethod
    def from_state_list(cls, state_list : StateList, data : Vector) -> "StateTable":
        return cls(
            data=np.array(data) if not isinstance(data, np.ndarray) else data,
            table_index=TableIndex(
                field_names=("state", ),
                field_domains=(state_list, )
            )
        )
    @classmethod
    def from_dict(cls, state_values : Mapping[State, float]) -> "StateTable":
        state_list, data = zip(*state_values.items())
        return cls.from_state_list(state_list, data)
    @property
    def state_list(self):
        return self.table_index.field_domains[0]

class StateActionTable(StateTable):
    @classmethod
    def from_state_action_lists(cls, state_list : StateList, action_list : ActionList, data : Matrix):
        return cls(
            data=np.array(data) if not isinstance(data, np.ndarray) else data,
            table_index=TableIndex(
                field_names=("state", "action"),
                field_domains=(state_list, action_list)
            )
        )
    @classmethod
    def from_state_list(cls, state_list : StateList, values : Vector):
        raise NotImplementedError
    @property
    def action_list(self):
        return self.table_index.field_domains[1]

class StateActionNextStateTable(StateActionTable):
    @classmethod
    def from_state_action_lists(cls, state_list : StateList, action_list : ActionList, data : Sequence[Matrix]):
        return cls(
            data=np.array(data) if not isinstance(data, np.ndarray) else data,
            table_index=TableIndex(
                field_names=("state", "action", "next_state"),
                field_domains=(state_list, action_list, state_list)
            )
        )

class StateNextStateTable(StateTable):
    @classmethod
    def from_state_list(cls, state_list : StateList, data : Matrix):
        return cls(
            data=np.array(data) if not isinstance(data, np.ndarray) else data,
            table_index=TableIndex(
                field_names=("state", "next_state"),
                field_domains=(tuple(state_list), tuple(state_list))
            )
        )