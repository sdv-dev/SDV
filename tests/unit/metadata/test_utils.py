from sdv.metadata.utils import cast_to_iterable, strings_from_regex


def test_cast_to_iterable():
    """Test ``cast_to_iterable``.

    Test that ``cast_to_iterable`` converts a signle object into a ``list`` but does not convert
    a ``list`` into a list inside a list.
    """
    # Setup
    value = 'abc'
    list_value = ['ab']

    # Run
    value = cast_to_iterable(value)
    list_value = cast_to_iterable(list_value)

    # Assert
    assert value == ['abc']
    assert list_value == ['ab']


def test_strings_from_regex_literal():
    generator, size = strings_from_regex('abcd')

    assert size == 1
    assert list(generator) == ['abcd']


def test_strings_from_regex_digit():
    generator, size = strings_from_regex('[0-9]')

    assert size == 10
    assert list(generator) == ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def test_strings_from_regex_repeat_literal():
    generator, size = strings_from_regex('a{1,3}')

    assert size == 3
    assert list(generator) == ['a', 'aa', 'aaa']


def test_strings_from_regex_repeat_digit():
    generator, size = strings_from_regex(r'\d{1,3}')

    assert size == 1110

    strings = list(generator)
    assert strings[0] == '0'
    assert strings[-1] == '999'
