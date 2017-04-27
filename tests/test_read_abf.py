import neurphys.read_abf as read_abf


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
    assert read_abf._all_ints([43.21, 12.34]) == False
    assert read_abf._all_ints(('test', '12345'))  == False
    assert read_abf._all_ints(('test', 12345)) == False
    assert read_abf._all_ints((54321, 12345))
    assert read_abf._all_ints((43.21, 12.34)) == False
