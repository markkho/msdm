# `msdm`: Models of Sequential Decision-Making

## Goals
`msdm` aims to simplify the design and evaluation of
models of sequential decision-making. The library
can be used for cognitive science or computer
science research/teaching.

## Approach
`msdm` provides standardized interfaces and implementations
for common constructs in sequential
decision-making. This includes algorithms used in single-agent
[reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) as well as those used in
[planning](https://en.wikipedia.org/wiki/Automated_planning_and_scheduling),
[partially observable environments](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process),
and [multi-agent games](https://en.wikipedia.org/wiki/Stochastic_game).

The library is organized around different **problem classes**
and **algorithms** that operate on **problem instances**.
We take inspiration from existing libraries such as
[scikit-learn](https://scikit-learn.org/) that
enable users to transparently mix and match components.
For instance, a standard way to define a problem, solve it,
and examine the results would be:

```
# create a problem instance
mdp = make_russell_norvig_grid(
    discount_rate=0.95,
    slip_prob=0.8,
)

# solve the problem
vi = ValueIteration()
res = vi.plan_on(mdp)

# print the value function
print(res.V)
```

The library is under active development. Currently, it primarily
supports tabular and discrete methods for single-agent problems,
but we aim to extend support for the following problem classes:

- Markov Decision Processes (MDPs)
- Partially Observable Markov Decision Processes (POMDPs)
- Markov Games
- Partially Observable Stochastic Games (POSGs)

and algorithms:

- Planning and Search
    - Value Iteration
    - Policy Iteration
    - Breadth-First Search
    - A*
    - Real-time Dynamic Programming
    - LAO*
- Reinforcement Learning
    - Q-Learning
    - Double Q-Learning
    - SARSA
    - Expected SARSA
- Inverse Reinforcement Learning
- Multi-agent learning
    - Correlated Q Learning
    - Nash Q Learning
    - Friend/Foe Q Learning

# Installation

## Installing from GitHub
```bash
$ pip install --upgrade git+https://github.com/markkho/msdm.git
```

## Installing the package in edit mode

After downloading, go into the folder and install the package locally
(with a symlink so its updated as source file changes are made):

```bash
$ pip install -e .
```

It is recommended to use a virtual environment.

Related libraries:
- [BURLAP](https://github.com/jmacglashan/burlap)

# Contributing

To run all tests: `make test`

To run tests for some file: `python -m py.test msdm/tests/$TEST_FILE_NAME.py`

To lint the code: `make lint`
