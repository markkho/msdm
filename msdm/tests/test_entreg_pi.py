import torch
from msdm.algorithms import PolicyIteration
from msdm.domains import GridWorld
from msdm.algorithms.entregpolicyiteration import \
    entropy_regularized_policy_iteration, \
    EntropyRegularizedPolicyIteration

def test_entreg_policy_iteration():
    gw = GridWorld(
        tile_array=[
            '..g',
            '...',
            '.##',
            '...',
            '##.',
            's..'
        ],
        step_cost=-1,
        discount_rate=.99
    )
    hard_res = PolicyIteration().plan_on(gw)
    tf = torch.from_numpy(gw.transition_matrix)
    tf.requires_grad = True
    rf = torch.from_numpy(gw.reward_matrix)
    ent_reg_params = dict(
        transition_matrix=tf,
        reward_matrix=rf,
        discount_rate=gw.discount_rate,
        n_planning_iters=100,
        policy_prior=None,
        initial_policy=None,
        check_convergence=True,
        q_value_range=None
    )
    high_ent_res = entropy_regularized_policy_iteration(
        **ent_reg_params,
        entropy_weight=5
    )
    med_ent_res = entropy_regularized_policy_iteration(
        **ent_reg_params,
        entropy_weight=1
    )
    low_ent_res = entropy_regularized_policy_iteration(
        **ent_reg_params,
        entropy_weight=.1
    )
    verylow_ent_res = entropy_regularized_policy_iteration(
        **ent_reg_params,
        entropy_weight=1e-20
    )
    hard_q = torch.from_numpy(hard_res._qvaluemat)
    re_s = torch.from_numpy(gw.reachable_state_vec)

    high_ent_diff = ((high_ent_res.action_values - hard_q)*re_s[:,None]).abs().sum().item()
    med_ent_diff = ((med_ent_res.action_values - hard_q)*re_s[:,None]).abs().sum().item()
    low_ent_diff = ((low_ent_res.action_values - hard_q)*re_s[:,None]).abs().sum().item()
    verylow_ent_diff = ((verylow_ent_res.action_values - hard_q)*re_s[:,None]).abs().sum().item()
    assert high_ent_diff > med_ent_diff > low_ent_diff > verylow_ent_diff

    # this should converge to the hard q-value
    assert torch.all(torch.isclose(verylow_ent_res.action_values*re_s[:,None], hard_q*re_s[:,None]))

    # we should be able to take the gradient through this
    assert tf.grad is None
    low_ent_res.action_values.sum().backward()
    assert tf.grad is not None
    assert torch.all(~torch.isnan(tf.grad))

    # policy prior - put a prior in staying in place
    pi0 = torch.tensor([[.1, .1, .1, .1, .1]])
    pi0[0, [(a['dx'], a['dy']) for a in gw.action_list].index((0, 0))] = .6
    med_ent_pi0_res = entropy_regularized_policy_iteration(
        **{
            **ent_reg_params,
            'policy_prior': pi0
        },
        entropy_weight=1
    )
    # the version using the stay-in-place prior should deviate less at all states
    # than the (default) version with a uniform prior over actions
    med_ent_pi0_dev = (torch.log(pi0/med_ent_pi0_res.policy)*pi0).sum(-1)
    med_ent_dev = (torch.log(pi0/med_ent_res.policy)*pi0).sum(-1)
    assert torch.all(med_ent_pi0_dev < med_ent_dev)

    # test wrapper
    EntropyRegularizedPolicyIteration().plan_on(gw)
