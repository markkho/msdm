from msdm.core.distributions.jointprobabilitytable import \
    Assignment, ConflictingKeyError, JointProbabilityTable, \
    UnnormalizedDistributionError, InconsistentVariablesError
from msdm.core.distributions.factors import make_factor

def test_consistent_assignments():
    pt = JointProbabilityTable.from_pairs([
        [dict(a=1, b=2), .5],
        [dict(a=1), .5],
    ])
    try:
        pt._check_valid()
        assert False
    except InconsistentVariablesError:
        pass

def test_unnormalized_distribution():
    pt = JointProbabilityTable.from_pairs([
        [dict(a=1, b=2), .9],
        [dict(a=1, b=3), .2],
    ])
    try:
        pt._check_valid()
        assert False
    except UnnormalizedDistributionError:
        pass

def test_marginalization_with_from_pairs():
    pt1 = JointProbabilityTable.from_pairs([
        [dict(a=1, b=2), .3],
        [dict(a=1, b=2), .3],
        [dict(a=1, b=3), .4],
    ])
    pt2 = JointProbabilityTable({
        Assignment.from_kwargs(a=1, b=3): .4,
        Assignment.from_kwargs(a=1, b=2): .6,
    })
    assert pt1 == pt2

def test_null_table():
    pt = JointProbabilityTable.from_pairs([
        [dict(a=1, b=2), .3],
        [dict(a=1, b=2), .3],
        [dict(a=1, b=3), .4],
    ])
    assert pt.join(JointProbabilityTable.null_table()) == pt
    assert id(pt.join(JointProbabilityTable.null_table())) != id(pt)
    @make_factor(use_cache=True, debug_mode=True)
    def null_factor() -> []:
        return None
    assert pt.join(null_factor) == pt

def test_join_with_no_shared_variables(p=.52, q=.92):
    pt1 = JointProbabilityTable.from_pairs([
        [dict(a=0), p],
        [dict(a=1), 1-p],
    ])
    pt2 = JointProbabilityTable.from_pairs([
        [dict(b=0), q],
        [dict(b=1), 1 - q],
    ])
    pt12 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), p*q],
        [dict(a=0, b=1), p*(1-q)],
        [dict(a=1, b=0), (1-p)*q],
        [dict(a=1, b=1), (1-p)*(1-q)],
    ])
    assert pt1.join(pt2).isclose(pt12, rtol=1e-12, atol=1e-12)

def test_join_with_subset_of_variables(p=.13, q=.87):
    pt1 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), p/2],
        [dict(a=0, b=1), p/2],
        [dict(a=1, b=0), (1-p)/2],
        [dict(a=1, b=1), (1-p)/2],
    ])
    pt2 = JointProbabilityTable.from_pairs([
        [dict(b=0), q],
        [dict(b=1), 1-q],
    ])
    pt12 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), p*q],
        [dict(a=0, b=1), p*(1-q)],
        [dict(a=1, b=0), (1-p)*q],
        [dict(a=1, b=1), (1-p)*(1-q)],
    ])
    assert pt1.join(pt2).normalize().isclose(pt12, rtol=1e-12, atol=1e-12), \
        (list(pt1.join(pt2).probs), list(pt12.probs))

def test_joint_probability_dist_with_factor(p=.13, q=.87):
    pa = JointProbabilityTable.from_pairs([
        [dict(a=0), p],
        [dict(a=1), 1-p],
    ])
    pb_a = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), q/2],
        [dict(a=0, b=1), (1 -q)/2],
        [dict(a=1, b=0), (1 -q)/2],
        [dict(a=1, b=1), q/2],
    ])
    @make_factor(use_cache=False, debug_mode=True)
    def func_b_a(a) -> ['b']:
        return JointProbabilityTable.from_pairs([
            [dict(b=a), q],
            [dict(b=1-a), 1 - q]
        ])
    join_pab = pa.join(pb_a).normalize()
    then_pab = pa.join(func_b_a).normalize()
    assert join_pab.isclose(then_pab)

def test_joint_probability_dist_multiple_joins(p=.91, q=.97):
    # mixture of experts A->B<-C
    pac = JointProbabilityTable.from_pairs([
        [dict(a=0, c=0), 1/4],
        [dict(a=0, c=1), 1/4],
        [dict(a=1, c=0), 1/4],
        [dict(a=1, c=1), 1/4],
    ])
    @make_factor(use_cache=False, debug_mode=True)
    def pb_a(a) -> ['b']:
        if a == 1:
            return JointProbabilityTable.from_pairs([
                [dict(b=0), 1 - p],
                [dict(b=1), p],
            ])
        return JointProbabilityTable.from_pairs([
            [dict(b=0), .5],
            [dict(b=1), .5],
        ])

    @make_factor(use_cache=False, debug_mode=True)
    def pb_c(c) -> ['b']:
        if c == 1:
            return JointProbabilityTable.from_pairs([
                [dict(b=0), 1 - q],
                [dict(b=1), q],
            ])
        return JointProbabilityTable.from_pairs([
            [dict(b=0), .5],
            [dict(b=1), .5],
        ])
    assert pac.join(pb_c).join(pb_a) == pac.join(pb_a).join(pb_c), \
        "single `join` should be order invariant"
    assert pac.join(pb_a, pb_c) == pac.join(pb_c, pb_a), \
        "grouped `join` should be order invariant"
    assert pac.join(pb_c, pb_a) == pac.join(pb_c).join(pb_a), \
        "`join` should be grouped/single invariant"
    pabc = pac.join(pb_a, pb_c).normalize()

    pabc_exp = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0, c=0), .5*.5],
        [dict(a=0, b=1, c=0), .5*.5],
        [dict(a=0, b=0, c=1), .5*(1-q)],
        [dict(a=0, b=1, c=1), .5*q],
        [dict(a=1, b=0, c=0), (1-p)*.5],
        [dict(a=1, b=1, c=0), p*.5],
        [dict(a=1, b=0, c=1), (1-p)*(1-q)],
        [dict(a=1, b=1, c=1), p*q],
    ]).normalize()
    assert pabc_exp.isclose(pabc)

def test_join_with_overlapping_variables(p=.5, q=.1):
    pt1 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), p/2],
        [dict(a=0, b=1), p/2],
        [dict(a=1, b=0), (1-p)/2],
        [dict(a=1, b=1), (1-p)/2],
    ])
    pt2 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), q/2],
        [dict(a=0, b=1), (1-q)/2],
        [dict(a=1, b=0), q/2],
        [dict(a=1, b=1), (1-q)/2],
    ])
    pt12 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), p*q],
        [dict(a=0, b=1), p*(1-q)],
        [dict(a=1, b=0), (1-p)*q],
        [dict(a=1, b=1), (1-p)*(1-q)],
    ])
    assert pt1.join(pt2).normalize().isclose(pt12, rtol=1e-12, atol=1e-12), \
        (list(pt1.join(pt2).probs), list(pt12.probs))

def test_common_cause_example(prob_ab=.88, prob_ac=.77):
    # a is a common cause of b and c, but a is uniform
    # b and c are unconditionally independent, but not conditionally independent
    def prob(a, b, c):
        return (prob_ab if a == b else 1-prob_ab)*(prob_ac if a == c else 1-prob_ac)
    pt1 = JointProbabilityTable.from_pairs([
        [dict(a=0, b=0), prob_ab/2],
        [dict(a=0, b=1), (1-prob_ab)/2],
        [dict(a=1, b=0), (1-prob_ab)/2],
        [dict(a=1, b=1), prob_ab/2],
    ])
    pt2 = JointProbabilityTable.from_pairs([
        [dict(a=0, c=0), prob_ac/2],
        [dict(a=0, c=1), (1-prob_ac)/2],
        [dict(a=1, c=0), (1-prob_ac)/2],
        [dict(a=1, c=1), prob_ac/2],
    ])
    pt12 = JointProbabilityTable.from_pairs([
        [d, prob(**d)] for d in
        [
            dict(a=0, b=1, c=0),
            dict(a=0, b=1, c=1),
            dict(a=0, b=0, c=0),
            dict(a=0, b=0, c=1),
            dict(a=1, b=1, c=0),
            dict(a=1, b=1, c=1),
            dict(a=1, b=0, c=0),
            dict(a=1, b=0, c=1),
        ]
    ]).normalize()
    assert pt1.join(pt2).normalize().isclose(pt12), "Joining 3 variables failed"

    # P(B) == P(C)
    p_b = pt12.marginalize(lambda x: x['b'])
    p_c = pt12.marginalize(lambda x: x['c'])
    assert p_b.isclose(p_c), "Should be independent"

    # P(B | A = 0) != P(C | A = 0)
    p_b_a0 = pt12.condition(lambda x: x['a'] == 0).marginalize(lambda x: x['b'])
    p_c_a0 = pt12.condition(lambda x: x['a'] == 0).marginalize(lambda x: x['c'])
    assert not p_b_a0.isclose(p_c_a0), "Should not be conditionally independent"

    # P(B | A = 0) != P(C | A = 1)
    p_b_a1 = pt12.condition(lambda x: x['a'] == 1).marginalize(lambda x: x['b'])
    p_c_a1 = pt12.condition(lambda x: x['a'] == 1).marginalize(lambda x: x['c'])
    assert not p_b_a1.isclose(p_c_a1), "Should not be conditionally independent"

def test_Assignment():
    # equality and hashing
    x = Assignment((('x', 3), ('y', 2)))
    y = Assignment((('y', 2), ('x', 3)))
    assert x == y
    d = {}
    d[x] = 1
    d[y] = 2
    assert len(d) == 1

    # disjoint variables
    z = Assignment((('y2', 2), ('x2', 3)))
    xz = x+z
    assert x.compatible_with(z)
    assert len(x+z) == len(x) + len(z)

    # overlapping variables
    good_z = Assignment((('x', 3),))
    x + good_z
    assert x.compatible_with(good_z)
    bad_z = Assignment((('x', 4),))
    try: # this should throw a key conflict error
        x + bad_z
        assert False
    except ConflictingKeyError:
        pass

    # test repr
    assert x == eval(x.__repr__())
