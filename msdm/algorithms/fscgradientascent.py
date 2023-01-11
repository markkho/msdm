from msdm.core.pomdp import TabularPOMDP
from msdm.core.pomdp.finitestatecontroller import StochasticFiniteStateController
from msdm.core.algorithmclasses import Learns, Result
import torch
import numpy as np

def stochastic_fsc_policy_evaluation_exact(pomdp: TabularPOMDP, fsc_action, fsc_state, *, fsc_initial_state=None, dtype=torch.float64):
    '''
    This function evaluates the policy represented by a stochastic controller by solving the
    fundamental equation of the Markov chain induced by taking the cross-product of
    a stochastic finite-state controller and its corresponding POMDP.

    Notation closely follows that of:
    Meuleau et al. (1999). Solving POMDPs by Searching the Space of Finite Policies.
    https://arxiv.org/abs/1301.6720
    '''

    T = torch.tensor(pomdp.transition_matrix, dtype=dtype)
    O = torch.tensor(pomdp.observation_matrix, dtype=dtype)
    R = torch.tensor(pomdp.state_action_reward_matrix, dtype=dtype)
    s0 = torch.tensor(pomdp.initial_state_vec, dtype=dtype)

    # Number of states, actions, observations in the POMDP
    nactions, nstates, nobs = O.shape
    # Number of states in the controller
    ncontroller = fsc_action.shape[0]

    # Checking that our controller strategies are distributions.
    assert torch.allclose(fsc_action.sum(axis=-1), torch.ones(fsc_action.shape[:-1], dtype=dtype))
    assert torch.allclose(fsc_state.sum(axis=-1), torch.ones(fsc_state.shape[:-1], dtype=dtype))
    assert fsc_initial_state is None or np.isclose(fsc_initial_state.sum(axis=-1).item(), 1)

    # First, we ensure our controller observation strategy is conditional on actions.
    # We'd like a distribution p(n' | n, a, o), but we accept 3-dimensional distributions
    # that correspond to p(n' | n, o). We convert the former to the latter when it's passed.
    if len(fsc_state.shape) == 3:
        assert fsc_state.shape == (ncontroller, nobs, ncontroller)
        fsc_state = fsc_state[:, None, :, :].expand(-1, nactions, -1, -1)
    assert fsc_state.shape == (ncontroller, nactions, nobs, ncontroller)

    # This is the Markov chain for the cross product of the stochastic controller & POMDP
    # action -> a, observation -> o
    # state -> s, next state -> t
    # agent state -> n, next agent state -> m
    Tmu = torch.einsum('na,sat,ato,naom->nsmt', fsc_action, T, O, fsc_state).contiguous()
    assert torch.allclose(Tmu.sum(axis=(2, 3)), torch.ones((ncontroller, nstates), dtype=dtype))

    # Expected immediate reward for each (node, state) pair
    Cmu = fsc_action@R.T

    # This is the size of the state space from taking the cross product of the controller & POMDP
    crossprod = ncontroller * nstates

    # Solving the fundamental equation of the controller/POMDP cross product.
    occupancy = (torch.eye(crossprod, dtype=dtype) - pomdp.discount_rate * Tmu.view((crossprod, crossprod))).inverse()
    V = occupancy @ Cmu.view(crossprod)

    # Reshape the value function back from the space of the controller/POMDP cross product
    V = V.view((ncontroller, nstates))

    # Return early if we don't have a distribution over initial states.
    if fsc_initial_state is None:
        return Result(state_controller_value=V)

    # This is a departure from Meuleau 1999; their initial controller state
    # distribution is conditional on an initial observation. However, our
    # formalization of the POMDP has an observation function that's conditional
    # on action. We can surely implement their initial controller state distribution
    # for some classes of POMDP, but leave that out for now.
    state_value = fsc_initial_state @ V

    return Result(
        state_controller_value=V,
        state_value=state_value,
        expected_value=state_value@s0,
    )

class FSCGradientAscent(Learns):
    def __init__(
        self, *,
        # Number of states the controller should have.
        controller_state_count,
        iterations=5000,
        learning_rate=1e-1,
        # This parameter is the number of iterations that should pass between updates printed  out
        log_iteration_progress=None,
        optimizer=torch.optim.Adam,
        dtype=torch.float64,
        seed=None,
    ):
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.log_iteration_progress = log_iteration_progress
        self.iterations = iterations
        self.controller_state_count = controller_state_count
        self.optimizer = optimizer
        self.seed = seed or torch.randint(2**30, size=(1,)).item()

    def train_on(self, pomdp: TabularPOMDP):
        # Number of states, actions, observations in the POMDP
        nactions, nstates, nobs = pomdp.observation_matrix.shape
        # Number of states in the finite state controller.
        ncontroller = self.controller_state_count

        with torch.random.fork_rng():
            torch.random.manual_seed(self.seed)
            fsc_action_logit = torch.rand(ncontroller, nactions, requires_grad=True, dtype=self.dtype)
            fsc_state_logit = torch.rand(ncontroller, nactions, nobs, ncontroller, requires_grad=True, dtype=self.dtype)
            fsc_initial_state_logit = torch.rand(ncontroller, requires_grad=True, dtype=self.dtype)

        def value():
            return stochastic_fsc_policy_evaluation_exact(
                pomdp,
                fsc_action_logit.softmax(-1),
                fsc_state_logit.softmax(-1),
                fsc_initial_state=fsc_initial_state_logit.softmax(-1),
                dtype=self.dtype,
            )

        opt = self.optimizer([fsc_action_logit, fsc_state_logit, fsc_initial_state_logit], lr=self.learning_rate)

        for idx in range(self.iterations):
            opt.zero_grad()

            result = value()

            loss = -result.expected_value
            loss.backward()
            opt.step()

            if self.log_iteration_progress and ((idx+1) % self.log_iteration_progress) == 0:
                print(f'iteration={idx} value={result.expected_value:.02f}')

        return Result(
            value=value(),
            policy=StochasticFiniteStateController(
                pomdp,
                fsc_action_logit.softmax(-1),
                fsc_state_logit.softmax(-1),
                fsc_initial_state_logit.softmax(-1),
            ),
            controller_logit=Result(
                action=fsc_action_logit,
                state=fsc_state_logit,
                initial_state=fsc_initial_state_logit,
            ),
        )
