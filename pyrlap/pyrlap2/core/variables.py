from collections import namedtuple

TaskVariable = namedtuple("TaskVariable", ["name", "domain", "properties"])
TaskVariable.__str__ = lambda v: f"<'{v.name}'>"
TaskVariable.__repr__ = TaskVariable.__str__

State = namedtuple("State", ["variables", "values"])
Action = namedtuple("Action", ["variables", "values"])
Background = namedtuple("Background", ["variables", "values"])

#useful constants
isTerminal = TaskVariable("isTerminal", (True, False), ())
TERMINALSTATE = State(variables=(isTerminal, ), values=(True, ))
NOTHINGSTATE = State(variables=(), values=())