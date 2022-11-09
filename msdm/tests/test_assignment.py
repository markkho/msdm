import numpy as np
from msdm.core.assignment import DefaultAssignmentMap, AssignmentMap

np.seterr(divide='ignore')

def test_AssignmentMap_encode():
    m = AssignmentMap()
    keys = [
        'ñ',
        b'hi',
        [3, 4],
        (1, 2),
        {'hi': 3},
        3,
    ]
    for key in keys:
        # Testing setter
        m[key] = 1337
    # Making sure we can also list keys
    assert len(list(m.keys())) == len(keys)
    for el in m.keys():
        assert el in keys

def test_DefaultAssignmentMap():
    m = DefaultAssignmentMap(lambda: 3)
    assert m['number'] == 3
    m['number'] = 7
    assert m['number'] == 7
    del m['number']
    assert m['apples'] == 3
    
    m = DefaultAssignmentMap(lambda key: key * 2)
    assert m[3] == 6
    m[3] = 7
    assert m[3] == 7
    del m[3]
    assert m[3] == 6