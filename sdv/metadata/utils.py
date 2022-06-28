"""Tools to generate strings from regular expressions."""

import re
import string

import numpy as np

import sre_parse  # isort:skip


def _literal(character, max_repeat):
    del max_repeat
    return iter([chr(character)]), 1


def _in(options, max_repeat):
    generators = []
    sizes = []
    for op, args in options:
        generator, size = _GENERATORS[op](args, max_repeat)
        generators.append(generator)
        sizes.append(size)

    return (value for generator in generators for value in generator), np.sum(sizes)


def _range(options, max_repeat):
    del max_repeat
    min_value, max_value = options
    max_value += 1
    return (chr(value) for value in range(min_value, max_value)), max_value - min_value


def _any(options, max_repeat):
    del options
    del max_repeat
    return iter(string.printable), len(string.printable)


def _max_repeat(options, max_repeat):
    min_, max_, options = options
    if max_ == sre_parse.MAXREPEAT:
        max_ = max_repeat

    op, args = options[0]
    generator, size = _GENERATORS[op](args, max_repeat)

    generators = []
    sizes = []
    for repeat in range(min_, max_ + 1):
        if repeat:
            sizes.append(size ** repeat)
            repeat_generators = [
                (_GENERATORS[op](args, max_repeat)[0], op, args)
                for _ in range(repeat)
            ]
            generators.append(_from_generators(repeat_generators, max_repeat))

    return (
        value
        for generator in generators
        for value in generator
    ), np.sum(sizes) + int(min_ == 0)


def _category_chars(regex):
    return [char for char in string.printable if regex.match(char)]


_CATEGORIES = {
    sre_parse.CATEGORY_SPACE: _category_chars(re.compile(r'\s')),
    sre_parse.CATEGORY_NOT_SPACE: _category_chars(re.compile(r'\S')),
    sre_parse.CATEGORY_DIGIT: _category_chars(re.compile(r'\d')),
    sre_parse.CATEGORY_NOT_DIGIT: _category_chars(re.compile(r'\D')),
    sre_parse.CATEGORY_WORD: _category_chars(re.compile(r'\w')),
    sre_parse.CATEGORY_NOT_WORD: _category_chars(re.compile(r'\W')),
}


def _category(category, max_repeat):
    del max_repeat
    characters = _CATEGORIES[category]
    return iter(characters), len(characters)


_GENERATORS = {
    sre_parse.LITERAL: _literal,
    sre_parse.IN: _in,
    sre_parse.RANGE: _range,
    sre_parse.ANY: _any,
    sre_parse.MAX_REPEAT: _max_repeat,
    sre_parse.CATEGORY: _category,
}


def _from_generators(generators, max_repeat):
    previous = [None] + [next(generator) for generator, _, _ in generators[1:]]

    remaining = True
    while remaining:
        string = []
        for index, (generator, op, args) in enumerate(generators):
            remaining = True
            try:
                value = next(generator)
                string.append(value)
                previous[index] = value
                string.extend(previous[index + 1:])
                break
            except StopIteration:
                generator = _GENERATORS[op](args, max_repeat)[0]
                generators[index] = generator, op, args
                value = next(generator)
                previous[index] = value
                string.append(value)
                remaining = False

        if remaining:
            yield ''.join(reversed(string))


def strings_from_regex(regex, max_repeat=16):
    """Generate strings that match the given regular expression.

    The output is a generator that produces regular expressions that match
    the indicated regular expressions alongside an integer indicating the
    total length of the generator.

    WARNING: Subpatterns are currently not supported.

    Args:
        regex (str):
            String representing a valid python regular expression.
        max_repeat (int):
            Maximum number of repetitions to produce when the regular
            expression allows an infinte amount. Defaults to 16.

    Returns:
        tuple:
            * Generator that produces strings that match the given regex.
            * Total length of the generator.
    """
    parsed = sre_parse.parse(regex, flags=sre_parse.SRE_FLAG_UNICODE)
    generators = []
    sizes = []
    for op, args in reversed(parsed):
        if op != sre_parse.AT:
            generator, size = _GENERATORS[op](args, max_repeat)
            generators.append((generator, op, args))
            sizes.append(size)

    return _from_generators(generators, max_repeat), np.prod(sizes)
