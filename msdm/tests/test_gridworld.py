from msdm.domains import GridWorld

def test_feature_locations():
    gw = GridWorld([
        "cacg",
        "sabb"])
    fl = gw.feature_locations
    lf = gw.location_features
    fl2 = {}
    for l, f in lf.items():
        fl2[f] = fl2.get(f, []) + [l,]
    assert all(set(fl[f]) == set(fl2[f]) for f in fl.keys())

def test_reachability():
    gw = GridWorld([
        "....#...g",
        "....#....",
        "#####....",
        "s........",
    ])
    assert len(gw.reachable_states()) == 22 #includes terminal
