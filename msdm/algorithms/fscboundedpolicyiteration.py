import torch
import numpy as np

from msdm.core.problemclasses.pomdp import TabularPOMDP
from msdm.core.problemclasses.pomdp.finitestatecontroller import StochasticFiniteStateController
from msdm.core.algorithmclasses import Learns, Result

from msdm.algorithms.fscgradientascent import stochastic_fsc_policy_evaluation_exact

class Solvers(object):
    @classmethod
    def qpth(cls, *args, **kwargs):
        import qpth
        return Result(
            solution=qpth.qp.QPFunction(**kwargs)(*args).squeeze(),
        )
    @classmethod
    def qpth_cvxpy(cls, *args, **kwargs):
        return cls.qpth(*args, solver=qpth.qp.QPSolvers.CVXPY, check_Q_spd=False, **kwargs)
    @classmethod
    def cvxpy_lp(cls, Q, p, G, h, A, b, solver='ECOS', verbose=False):
        import cvxpy as cp
        # Making sure we can ignore Q
        assert (
            np.allclose(torch.zeros(Q.shape), Q) or
            np.allclose(torch.eye(Q.shape[0]), Q/Q[0, 0])
        )

        z = cp.Variable(len(p))
        prob = cp.Problem(
            cp.Minimize(p.T@z), [
                G@z <= h,
                A@z == b])
        solution = prob.solve(verbose=verbose, solver=getattr(cp, solver) if isinstance(solver, str) else solver)
        return Result(
            solution=torch.tensor(z.value, dtype=Q.dtype),
            problem=prob,
        )

def improve_node(
    pomdp, V, node, *,
    qcoef=1e-8,
    dtype=torch.float64,

    solver=Solvers.cvxpy_lp,
    cvxpy=True,
    solver_kwargs={},
):
    '''
    This implements a linear program to find a combination of backed-up FSC nodes that dominate
    an existing node. It implements the efficient version described in Table 4 of [1]. In particular,
    we follow the suggestion in a footnote to compute c_a based on the c_{a,n_z} values, leading to a
    total parameter size of |A||N||Z|+1.

    Since we compute c_a based on c_{a,n_z}, we can further simplify the LP in Table 4:
    - We require c_{a,n_z} sums to 1, replacing the constraints that they sum to c_a which in turn sums to 1.
    - We drop the constraint that c_a is non-negative; the constraint that c_{a,n_z} is non-negative is
      is sufficient to assure c_a will be as well.

    [1] Poupart, Boutilier. (2003). Bounded Finite State Controllers
    '''

    T = torch.tensor(pomdp.transition_matrix, dtype=dtype)
    O = torch.tensor(pomdp.observation_matrix, dtype=dtype)
    R = torch.tensor(pomdp.state_action_reward_matrix, dtype=dtype)

    # Number of states, actions, observations in the POMDP
    nactions, nstates, nobs = pomdp.observation_matrix.shape
    # Number of states in the finite state controller.
    ncontroller = V.shape[0]

    assert V.shape == (ncontroller, nstates)

    '''
    Making sizing variables for our parameters.

    Our parameters contains two types of variables, in the following order:
    - the c_{a,n_z} variables (accessed by canz_idxs)
    - the epsilon term (acessed by epsilon_idx)
    '''
    canz_shape = torch.Size((nactions, nobs, ncontroller))
    canz_count = canz_shape.numel()
    param_count = canz_count + 1
    canz_idxs = slice(0, canz_count)
    epsilon_idx = slice(canz_count, canz_count+1)
    # Just asserting that our indexes cover the full range of parameters.
    assert param_count == (canz_idxs.stop-canz_idxs.start) + (epsilon_idx.stop-epsilon_idx.start)

    '''
    Starting with the minimization objective: argmin_z 1/2 z^T Q z + p^T z
    We don't have a quadratic constraint here but one of the qpth solvers requires
    it be SPD, so I've included an identity matrix considerably scaled down. When we
    solve with cvxpy, we can let this be zero.

    The only variable we seek to optimize is epsilon; since we want to maximize it
    but we are specifying a minimization, we negate it.
    '''
    Q = torch.eye(param_count, dtype=dtype) * qcoef
    # The default optimization is an argmin, so this lets us maximize the epsilon.
    p = torch.zeros(param_count, dtype=dtype)
    p[epsilon_idx] = -1

    '''
    Now, the equality constraints: Az=b. We have only one; that \sum_{a,n_z} c_{a,n_z} = 1.
    '''
    A = torch.zeros((1, param_count), dtype=dtype)
    b = torch.zeros((1,), dtype=dtype)
    A[0, canz_idxs] = 1
    b[0] = 1

    '''
    Finally, the inequality constraints: Gz<=h.

    We encode two types of constraints (contiguously, in this order):
    - The value improvement constraint at each state. We shift things around to fit Gz<=h,
      as well as expand the c_a into a \sum_{n_z} c_{a,n_z}.
    - That each c_{a,n_z} are non-negative, c >= 0. To fit Gz<=h, we rearrange
      to -c <= 0, specified in G with a -1 for each c_{a,n_z}.
    '''
    G = torch.zeros((nstates + canz_count, param_count), dtype=dtype)
    h = torch.cat((
        # Have to negate value here, because it switches sides.
        -V[node],
        torch.zeros(canz_count, dtype=dtype),
    ))

    # Earlier locations encode value improvement constraint.
    for s in range(nstates):
        # Epsilon stays on left side, so positive coef.
        G[s, epsilon_idx] = 1
        # Doing some reshaping to make the rest of this a bit easier to read.
        canz_coef = G[s, canz_idxs].view(canz_shape)
        for a in range(nactions):
            for o in range(nobs):
                for nn in range(ncontroller):
                    # Have to negate backed-up value b/c it switches sides.
                    canz_coef[a, o, nn] = -(
                        # Since c_a = \sum_{n_z} c_{a,n_z}, we have to include the
                        # reward for each c_{a,n_z}.
                        R[s, a] +
                        # This follows pretty directly from Table 4; we take an expectation
                        # of the value function after marginalizing out next state.
                        pomdp.discount_rate * (T[s, a, :] * O[a, :, o] * V[nn]).sum())

    # Last locations encode c >= 0 as -c <= 0
    for idx in range(canz_count):
        G[nstates+idx, idx] = -1

    # Solve the LP
    constraints = (Q, p, G, h, A, b)
    result = solver(*constraints)

    # Unpack the solution.
    epsilon = result.solution[epsilon_idx].item()
    canz = result.solution[canz_idxs].view(canz_shape)
    # Computing c_a using c_{a,n_z}
    c_a = canz.sum(axis=(1, 2))
    assert c_a.shape == (nactions,)
    print(canz)
    assert ((canz >= 0) | np.isclose(canz, 0)).all()
    assert np.isclose(torch.sum(canz), 1)

    return Result(
        constraints=constraints,
        epsilon=epsilon,
        canz=canz,
        c_a=c_a,
        action_strategy=c_a,
        observation_strategy=canz/canz.sum(-1, keepdims=True),
        solver_result=result,
    )

class FSCBoundedPolicyIteration(Learns):
    def __init__(
        self, *,
        # Number of states the controller should have.
        controller_state_count,
        iterations=100,
        learning_rate=1e-1,
        # This parameter is the number of iterations that should pass between updates printed  out
        log_iteration_progress=None,
        optimizer=torch.optim.Adam,
        dtype=torch.float64,
        seed=None,
        convergence_diff=1e-4,
    ):
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.log_iteration_progress = log_iteration_progress
        self.iterations = iterations
        self.controller_state_count = controller_state_count
        self.optimizer = optimizer
        self.seed = seed or torch.randint(2**30, size=(1,)).item()
        self.convergence_diff = convergence_diff

    def train_on(self, pomdp: TabularPOMDP):
        # Our objective is to maximize the negative mean value
        objective=lambda V: V.mean()

        # Number of states, actions, observations in the POMDP
        nactions, nstates, nobs = pomdp.observation_matrix.shape
        # Number of states in the finite state controller.
        ncontroller = self.controller_state_count

        with torch.random.fork_rng():
            torch.random.manual_seed(self.seed)
            fsc_action = torch.rand(ncontroller, nactions, dtype=self.dtype).softmax(-1)
            fsc_state = torch.rand(ncontroller, nactions, nobs, ncontroller, dtype=self.dtype).softmax(-1)
            fsc_initial_state = torch.rand(ncontroller, dtype=self.dtype).softmax(-1)

        # HACK is expected_value right? HACK is initial state right?
        value = lambda: stochastic_fsc_policy_evaluation_exact(
            self.pomdp, fsc_action, fsc_state, fsc_initial_state, dtype=self.dtype).state_controller_value

        V = value()
        converged = False
        for idx in range(self.iterations):
            prev = V.clone()
            for n in range(ncontroller):
                V = value()
                constraints, unpack = make_constraints_simple(n, V, T, O, R, gamma, dtype=self.dtype, qcoef=1e-10)
                solution = qpth.qp.QPFunction()(*constraints).float()
                epsilon, c_a, canz = unpack(solution[0])

                prev_fsc = (fsc_action, fsc_state)
                # Have to clone so we can take a gradient through the below modifications.
                fsc_action, fsc_state = fsc_action.clone(), fsc_state.clone()
                fsc_action[n] = c_a
                # Normalizing by transition distribution to get the conditional distribution p(n'|n,a,o)
                #fsc_state[n] = canz/canz.sum(-1)[:, :, None]
                fsc_state[n] = canz/canz.sum(-1, keepdims=True)
                Vafter = value()

                # HACK wonder if we can take a gradient through this??
                if rollback_bad_change and objective(V) > objective(Vafter):
                    fsc_action, fsc_state = prev_fsc

            V = value()

            if self.log_iteration_progress and ((idx+1) % self.log_iteration_progress) == 0:
                print(idx, round(objective(V).item(), 3))

            if np.abs(prev-V).max() < self.convergence_diff:
                converged = True
                break

        return Result(
            converged=converged,
            policy=StochasticFiniteStateController(pomdp, fsc_action, fsc_state, fsc_initial_state),
            # xxx?
            V=V,
            value=objective(V),
        )
