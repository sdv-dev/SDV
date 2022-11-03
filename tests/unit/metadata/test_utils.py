from sdv.metadata.utils import strings_from_regex


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
