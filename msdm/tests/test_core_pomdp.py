import numpy as np
from msdm.domains.tiger import Tiger

def assert_dist_vec_match(item_list, dist, vec):
    for i, x in enumerate(item_list):
        assert np.isclose(dist.prob(x), vec[i])
def state_dist_to_vec(p, dist):
    return np.array([dist.prob(p.state_list[si]) for si in range(len(p.state_list))])

def test_state_estimator():
    p = Tiger(coherence=0.85, discount_rate=0.95)

    # Generating a list of belifs
    b = p.initial_state_dist()
    beliefs = [b]
    for _ in range(4):
        b = p.state_estimator(b, 'listen', 'left')
        beliefs.append(b)

    # For each of these beliefs, we make sure both ways of computing the posterior are right.
    for belief in beliefs:
        for ai, a in enumerate(p.action_list):
            for oi, o in enumerate(p.observation_list):
                belief_vec = state_dist_to_vec(p, belief)
                posterior = p.state_estimator(belief, a, o)
                posterior_vec = p.state_estimator_vec(belief_vec, ai, oi)
                assert_dist_vec_match(p.state_list, posterior, posterior_vec)

                # Just computing it explicitly here too
                posterior_vec2 = (belief_vec @ p.transition_matrix[:, ai, :]) * p.observation_matrix[ai, :, oi]
                posterior_vec2 /= posterior_vec2.sum()
                assert_dist_vec_match(p.state_list, posterior, posterior_vec2)

def test_predictive_observation():
    p = Tiger(coherence=0.85, discount_rate=0.95)

    # Generating a list of belifs
    b = p.initial_state_dist()
    beliefs = [b]
    for _ in range(4):
        b = p.state_estimator(b, 'listen', 'left')
        beliefs.append(b)

    # For each of these beliefs, we make sure both ways of computing the predictive are right.
    for belief in beliefs:
        for ai, a in enumerate(p.action_list):
            for oi, o in enumerate(p.observation_list):
                belief_vec = state_dist_to_vec(p, belief)
                p_dist = p.predictive_observation_dist(belief, a)
                p_vec = p.predictive_observation_vec(belief_vec, ai)
                assert_dist_vec_match(p.observation_list, p_dist, p_vec)
