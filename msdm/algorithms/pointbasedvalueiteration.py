"""Point-based Value Iteration

Based on Pineau, Gordon & Thrun (2003). Point-based Value Iteration: An Anytime Algorithm for POMDPs.

Also see Shani, Pineau & Kaplow (2012) A survey of point-based POMDP solvers

"""
import numpy as np
from scipy.spatial.distance import cdist

from msdm.core.algorithmclasses import Plans, Result
from msdm.core.pomdp import TabularPOMDP
from msdm.core.pomdp.alphavectorpolicy import AlphaVectorPolicy
from msdm.core.pomdp.pomdp import PartiallyObservableMDP

def next_beliefs(pomdp : PartiallyObservableMDP, b):
    assert np.isclose(sum(b), 1), sum(b)
    tf = pomdp.transition_matrix
    of = pomdp.observation_matrix
    ns_dist = np.einsum("san,s->an", tf, b)
    nbs = np.einsum("an,ano->aon", ns_dist, of)
    nbs = nbs[nbs.sum(-1) > 0, :] #ignore 0 beliefs
    nbs = nbs/nbs.sum(-1, keepdims=True)
    nbs = nbs.reshape((-1, nbs.shape[-1]))
    return np.unique(nbs, axis=0)

def expand_beliefs(pomdp : PartiallyObservableMDP, belief_set):
    new_bs = []
    for b in belief_set:
        # First generate all beliefs following from this one
        nbs = next_beliefs(pomdp, b)

        # only add the belief that is the furthest from existing beliefs
        # this is the heuristic used in Pineau et al 2003
        # note we add in beliefs that are tied in distance to ensure symmetry
        nbs_bs_dist = cdist(nbs, belief_set)
        L1_nb_dist = nbs_bs_dist.min(axis=1)
        max_L1_nb_dist = np.max(L1_nb_dist)
        if max_L1_nb_dist == 0.0: # no beliefs are new
            continue
        max_L1_nbs = nbs[L1_nb_dist == max_L1_nb_dist]
        new_bs.append(max_L1_nbs)
    new_bs = np.concatenate(new_bs)
    belief_set = np.concatenate((belief_set, new_bs))
    belief_set = np.unique(belief_set, axis=0)
    return belief_set

def point_based_value_iteration(
    pomdp : PartiallyObservableMDP,
    belief_set,
    value_convergence_epsilon,
    horizon=None
):
    # iterations for infinite horizon as suggested in Pineau et al. 2003
    if horizon is None:
        rmax = pomdp.state_action_reward_matrix.max().item()
        rmin = pomdp.state_action_reward_matrix.min().item()
        horizon = value_convergence_epsilon / (rmax - rmin)
        horizon = np.log(horizon) / np.log(pomdp.discount_rate)
        horizon = int(np.ceil(horizon))

    tf = pomdp.transition_matrix
    sa_rf = pomdp.state_action_reward_matrix
    nt = ~pomdp.absorbing_state_vec.astype(bool)
    of = pomdp.observation_matrix
    aa = pomdp.action_list
    ss = pomdp.state_list
    oo = pomdp.observation_list

    bb = belief_set
    count_b = np.arange(len(bb))

    sa_rf = sa_rf*nt[:,None] #reward at terminal state is 0
    tf = tf*nt[:, None, None] #terminal states transition nowhere

    # alpha vectors - one per belief
    bv = np.zeros((len(bb), len(pomdp.state_list)))

    for i in range(horizon):
        ### Alpha-vectors over states (s) associated with each action (a),
        ### observation (o), and next-belief (p)
        aops_fut_vf = np.einsum("san,ano,pn->aops", tf, of, bv)

        ### We want to find the indices of the best next-belief (p) alpha-vector
        ### for every last-belief (b), action (a), and observation (o)
        aobp_fut_vf = np.einsum("aops,bs->aobp", aops_fut_vf, bb)
        xp_fut_vf = aobp_fut_vf.reshape((len(aa) * len(oo) * len(bb), -1))
        xp_fut_vf_max_idx = xp_fut_vf.argmax(axis=1)

        ### Now we have the indices, extract the best next-belief (p)
        ### alpha-vector from the current value function and reshape
        ### so that the first 3 dimensions are action, observation, and
        ### last-belief (next-belief, p is now implicit)
        xn_alpha_star = bv[xp_fut_vf_max_idx, :]
        aobn_alpha_star = xn_alpha_star.reshape((len(aa), len(oo), len(bb), len(ss)))

        ### Recalculate the next-state alpha-vectors over states associated with each
        ### action, observation, last-belief
        aobs_fut_vf = np.einsum("san,ano,aobn->aobs", tf, of, aobn_alpha_star)

        ### Now we can marginalize out the observations (summation in Eq 9)
        abs_fut_vf = np.einsum("aobs->abs", aobs_fut_vf)

        ### Combine the one-step reward with the next step value
        bsa_fut_vf = np.einsum("abs->bsa", abs_fut_vf)
        bsa_vf = sa_rf[None, :, :] + pomdp.discount_rate * bsa_fut_vf

        ### Calculate the best action alpha-vec out of the set of
        ### action alpha-vecs associated with each belief
        ba_vf = np.einsum("bsa,bs->ba", bsa_vf, bb)
        ba_vf_max_idx = ba_vf.argmax(axis=1)
        new_bv = bsa_vf[count_b, :, ba_vf_max_idx]

        # convergence test
        old_v = np.einsum("bs,bs->b", bv, bb)
        new_v = np.einsum("bs,bs->b", new_bv, bb)
        delta = np.abs(old_v - new_v).max()
        if delta < value_convergence_epsilon:
            break
        bv = new_bv
    return {
        'alpha_vectors': bv,
        'belief_action_alpha_vectors': bsa_vf,
        'belief_action_indices': ba_vf_max_idx,
        'iterations': i
    }

def belief_values(alpha_vectors, belief_set):
    return np.max(np.einsum("bs,ds->db", alpha_vectors, belief_set), axis=-1)

class PointBasedValueIteration(Plans):
    def __init__(
        self,
        min_belief_expansions=int(1e2),
        max_belief_expansions=int(1e5),
        value_convergence_epsilon=.01,
        horizon=None
    ):
        """
        Point-based value iteration approximates an exact
        POMDP value iteration solution by selecting a
        small set of representative belief points and then tracking
        the value and its derivative for those points only.
        This implementation is based on Pineau et al. (2003).

        Parameters
        ----------
        min_belief_expansions : int
            The minimum number of iterations of belief-set expansions.
            At each iteration, for each belief in the current belief set
            we find the furthest successor belief and add it to the
            set.
        max_belief_expansions : int
            The maximum number of belief set expansions.
        value_convergence_epsilon : float
            The convergence crition used for point-based value iteration
            (inner loop) as well as belief set expansions (outer loop).
        horizon : int
            The planning horizon to optimize value over.
            None corresponds to an infinite horizon.
            If this is not None, then it overrides value_convergence_epsilon.

        Returns
        -------
        Result
            A result object with the computed policy
        """
        self.min_belief_expansions = min_belief_expansions
        self.max_belief_expansions = max_belief_expansions
        self.value_convergence_epsilon = value_convergence_epsilon
        self.horizon = horizon

    def _solve(self, pomdp):
        s0 = pomdp.initial_state_vec
        belief_set = np.array([s0,])
        iterator = range(self.max_belief_expansions)
        for i in iterator:
            belief_set = expand_beliefs(pomdp, belief_set)
            if i >= self.min_belief_expansions:
                break

        last_res = None
        for i in iterator:
            # run value iteration to convergence
            res = point_based_value_iteration(
                pomdp,
                belief_set,
                value_convergence_epsilon=self.value_convergence_epsilon,
                horizon=self.horizon
            )

            # expand belief set
            belief_set = expand_beliefs(pomdp, belief_set)

            # convergence check
            if last_res:
                last_v = belief_values(last_res['alpha_vectors'], belief_set)
                curr_v = belief_values(res['alpha_vectors'], belief_set)
                diff = np.max(np.abs(last_v - curr_v))
                if diff < self.value_convergence_epsilon:
                    break
            last_res = res
        del res['iterations']
        res['belief_set'] = belief_set
        res['expansion_iterations'] = i + self.min_belief_expansions
        return res

    def plan_on(self, pomdp: TabularPOMDP):
        res = self._solve(pomdp)
        pi = AlphaVectorPolicy(pomdp, res['alpha_vectors'])
        return Result(
            policy=pi,
            alpha_vectors=res['alpha_vectors'],
            alpha_actions=[pomdp.action_list[i] for i in res['belief_action_indices']],
            belief_set=res['belief_set'],
            expansion_iterations=res['expansion_iterations']
        )
