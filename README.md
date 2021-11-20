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
[reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
as well as those used in
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

The library is under active development. Currently,
we support the following problem classes:

- Markov Decision Processes (MDPs)
- Partially Observable Markov Decision Processes (POMDPs)
- Markov Games
- Partially Observable Stochastic Games (POSGs)

The following algorithms have been implemented and
tested:

- Classical Planning
    - Breadth-First Search (Zuse, 1945)
    - A* (Hart, Nilsson & Raphael, 1968)
- Stochastic Planning
    - Value Iteration (Bellman, 1957)
    - Policy Iteration (Howard, 1960)
    - Labeled Real-time Dynamic Programming ([Bonet & Geffner, 2003](https://www.aaai.org/Papers/ICAPS/2003/ICAPS03-002.pdf))
    - LAO* ([Hansen & Zilberstein, 2003](https://www.sciencedirect.com/science/article/pii/S0004370201001060))
- Partially Observable Planning
    - QMDP ([Littman, Cassandra & Kaelbling, 1995](https://www.sciencedirect.com/science/article/pii/B9781558603776500529))
    - Point-based Value-Iteration ([Pineau, Gordon & Thrun, 2003](https://dl.acm.org/doi/abs/10.5555/1630659.1630806))
    - Finite state controller gradient ascent ([Meuleau, Kim, Kaelbling & Cassandra, 1999](https://arxiv.org/abs/1301.6720))
    - Bounded finite state controller policy iteration ([Poupart & Boutilier, 2003](https://dl.acm.org/doi/abs/10.5555/2981345.2981448))
    - Wrappers for [POMDPs.jl](https://juliapomdp.github.io/POMDPs.jl/latest/) solvers (requires Julia installation)
- Reinforcement Learning
    - Q-Learning (Watkins, 1992)
    - Double Q-Learning ([van Hasselt, 2010](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html))
    - SARSA ([Rummery & Niranjan, 1994](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf))
    - Expected SARSA ([van Seijen, van Hasselt, Whiteson & Wiering, 2009](https://ieeexplore.ieee.org/abstract/document/4927542))
- Multi-agent Reinforcement Learning (in progress)
    - Correlated Q Learning ([Greenwald & Hall, 2002](https://dl.acm.org/doi/abs/10.5555/3041838.3041869))
    - Nash Q Learning ([Hu & Wellman, 2003](https://dl.acm.org/doi/abs/10.5555/945365.964288))
    - Friend/Foe Q Learning ([Littman, 2001](https://dl.acm.org/doi/abs/10.5555/645530.655661))

We aim to add implementations for other algorithms in the
near future (e.g., inverse RL, deep learning, multi-agent learning and planning).

# Installation

It is recommended to use a [virtual environment](https://virtualenv.pypa.io/en/latest/index.html).

## Installing from pip

```bash
$ pip install msdm
```

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

# Contributing

We welcome contributions in the form of implementations of
algorithms for common problem classes that are
well-documented in the literature. Please first
post an issue and/or
reach out to <mark.ho.cs@gmail.com>
to check if a proposed contribution is within the
scope of the library.

## Running tests, etc.

To run all tests: `make test`

To run tests for some file: `python -m py.test msdm/tests/$TEST_FILE_NAME.py`

To lint the code: `make lint`
