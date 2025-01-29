def test_check_value_as_integer():
    assert check_value_as_integer(10) == True
    assert check_value_as_integer(3.14) == False
    assert check_value_as_integer("hello") == False
    assert check_value_as_integer(True) == False
    assert check_value_as_integer(None) == False