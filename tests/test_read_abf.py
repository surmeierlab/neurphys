import neurphys.read_abf as read_abf
import neurphys.utilities as util


def test_all_strs():
    assert read_abf._all_strs(['test', '12345'])
    assert read_abf._all_strs(['test', 12345]) == False
    assert read_abf._all_strs([54321, 12345]) == False
    assert read_abf._all_strs(('test', '12345'))
    assert read_abf._all_strs(('test', 12345)) == False
    assert read_abf._all_strs((54321, 12345)) == False


def test_all_ints():
    assert read_abf._all_ints(['test', '12345'])  == False
    assert read_abf._all_ints(['test', 12345]) == False
    assert read_abf._all_ints([54321, 12345])
    assert read_abf._all_ints(['4321', 12.34]) == False
    assert read_abf._all_ints(('test', '12345'))  == False
    assert read_abf._all_ints(('test', 12345)) == False
    assert read_abf._all_ints((54321, 12345))
    assert read_abf._all_ints(('4321', 12.34)) == False

# Testing of keep/drop functions is on hold until bugs have been worked out

# def test_keep_sweeps():
#     df = util.mock_multidf()
#     sweep_ints  = [1,4]
#     sweep_mixed = [1,4,5.23]
#     sweep_names = ['sweep002','sweep003']
#     sweep_wrong = ['sweepnop']
#     sweep_strnotsweep = ['thing']
#     sweep_disoredered = ['sweep003', 'sweep002']
