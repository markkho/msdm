from collections import defaultdict
from types import SimpleNamespace
from typing import Union
import torch
import numpy as np
import warnings
from msdm.core.mdp import TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

def clamp_zero(tensor):
    min_val = torch.finfo(tensor.dtype).tiny
    return torch.clamp(tensor, min=min_val)


def entropy_regularized_policy_iteration(
    transition_matrix: torch.Tensor,
    reward_matrix : torch.Tensor,
    discount_rate : float,
    entropy_weight : Union[torch.Tensor, float],
    n_planning_iters : int,
    policy_prior : torch.Tensor=None,
    initial_policy : torch.Tensor=None,
    check_convergence : bool=True,
    force_nonzero_probabilities: bool=True
):
    """
    An implementation of entropy regularized policy iteration
    (e.g., Geist, Scherrer, & Pietquin (2019) ICML). This takes in
    pytorch tensors and returns tensors that result from the computation.

    Parameters:
        transition_matrix: An SxAxS tensor representing MDP transition probabilities
        reward_matrix: An (1|S)x(1|A)x(1|S) tensor representing MDP rewards
        discount_rate: The discount rate
        entropy_weight: Weight(s) on the entropy regularization. When a S-length vector
            is passed in, each entry corresponds to the weight on that state.
        n_planning_iters: Iterations of policy iteration (improvement and evaluation)
        policy_prior: An (1|S)xA tensor representing policy prior probabilities
        initial_policy: An SxA tensor representing the initial policy for policy iteration
        check_convergence: Whether to check convergence
        force_nonzero_probabilities: Whether or not policies can have zero probabilities.

    Returns:
        A SimpleNamespace with the results of the algorithm.
    """
    tf = transition_matrix
    rf = reward_matrix
    if policy_prior is None:
        policy_prior = torch.softmax(torch.ones((1, tf.shape[1]), dtype=tf.dtype), 1)
    if initial_policy is None:
        initial_policy = torch.softmax(torch.ones(tf.shape[:-1], dtype=tf.dtype)*torch.log(policy_prior > 0), 1)
    assert isinstance(transition_matrix, torch.Tensor)
    assert initial_policy.shape == tf.shape[:-1]
    pi0 = clamp_zero(policy_prior) if force_nonzero_probabilities else policy_prior
    assert tf.shape[0] == tf.shape[2]
    assert rf.shape[0] in (1, tf.shape[0])
    assert rf.shape[1] in (1, tf.shape[1])
    assert rf.shape[2] in (1, tf.shape[2])
    assert pi0.shape[0] in (1, tf.shape[0])
    assert pi0.shape[1] == tf.shape[1]
    if isinstance(entropy_weight, (float, int)):
        entropy_weight = torch.tensor([entropy_weight])
    assert entropy_weight.shape[0] in (1, tf.shape[0])

    eye = torch.eye(tf.shape[0])
    pi = clamp_zero(initial_policy) if force_nonzero_probabilities else initial_policy
    converged = False
    for i in range(n_planning_iters):
        s_ent = torch.nansum(torch.log(pi/pi0)*pi, dim=1)
        s_rf = torch.einsum("san,san,sa->s",rf,tf,pi)
        s_rf_ent = (s_rf - entropy_weight*s_ent)
        mp = (pi[:,:,None]*tf[:, :, :]).sum(dim=1)
        v = torch.linalg.solve(eye - discount_rate*mp, s_rf_ent)
        q = (tf[:,:,:]*(rf + discount_rate*v[None,None,:])).sum(dim=-1)
        q_scale = q_action = (1/entropy_weight[:,None])*q
        new_pi = torch.softmax(q_action + torch.log(pi0), -1)
        if check_convergence:
            if torch.all(torch.isclose(pi, new_pi)):
                converged = True
                break
        pi = clamp_zero(new_pi) if force_nonzero_probabilities else new_pi
    return SimpleNamespace(
        policy=pi,
        action_values=q,
        state_rewards=s_rf,
        policy_entropy=s_ent,
        state_values=v,
        iterations=i,
        converged=converged
    )

class EntropyRegularizedPolicyIteration(Plans):
    def __init__(
        self,
        iterations=None,
        entropy_weight=1,
        policy_prior=None
    ):
        self.iterations = iterations
        self.entropy_weight = entropy_weight
        self.policy_prior = policy_prior

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        ss = mdp.state_list
        tf = torch.from_numpy(mdp.transition_matrix.copy())
        rf = torch.from_numpy(mdp.reward_matrix.copy())
        am = torch.from_numpy(mdp.action_matrix.copy())

        iterations = self.iterations
        if iterations is None:
            iterations = max(len(ss), int(1e5))

        if self.policy_prior is None:
            policy_prior = am/am.sum(-1, keepdims=True)
        else:
            policy_prior = self.policy_prior

        pi_res = entropy_regularized_policy_iteration(
            transition_matrix=tf,
            reward_matrix=rf,
            discount_rate=mdp.discount_rate,
            entropy_weight=self.entropy_weight,
            n_planning_iters=iterations,
            policy_prior=policy_prior,
            initial_policy=None,
            check_convergence=True,
            force_nonzero_probabilities=True
        )
        policy = TabularPolicy.from_state_action_lists(
            mdp.state_list,
            mdp.action_list,
            pi_res.policy.detach().numpy()
        )
        res = PlanningResult()
        res.converged = pi_res.converged
        if not res.converged:
            warnings.warn(f"Entropy Regularized Policy Iteration not converged after {pi_res.iterations} iterations")
        res.mdp = mdp
        res.policy = res.pi = policy
        res._valuevec = pi_res.state_values.detach().numpy()
        vf = dict()
        for s, vi in zip(mdp.state_list, res._valuevec):
            vf[s] = vi
        res.valuefunc = res.V = vf
        res._qvaluemat = pi_res.action_values.detach().numpy()
        res.iterations = pi_res.iterations
        qf = defaultdict(lambda : dict())
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                qf[s][a] = res._qvaluemat[si, ai]
        res.actionvaluefunc = res.Q = qf
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])
        res.policy_divergence = dict(zip(mdp.state_list, pi_res.policy_entropy))
        return res
