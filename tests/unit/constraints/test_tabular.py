"""Tests for the sdv.constraints.tabular module."""

import uuid

import numpy as np
import pandas as pd
import pytest

from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.tabular import (
    Between, ColumnFormula, CustomConstraint, GreaterThan, Negative, OneHotEncoding, Positive,
    Rounding, UniqueCombinations)


def dummy_transform():
    pass


def dummy_reverse_transform():
    pass


def dummy_is_valid():
    pass


class TestCustomConstraint():

    def test___init__(self):
        """Test the ``CustomConstraint.__init__`` method.

        The ``transform``, ``reverse_transform`` and ``is_valid`` methods
        should be replaced by the given ones, importing them if necessary.

        Setup:
        - Create dummy functions (created above this class).

        Input:
        - dummy transform and revert_transform + is_valid FQN
        Output:
        - Instance with all the methods replaced by the dummy versions.
        """
        is_valid_fqn = __name__ + '.dummy_is_valid'

        # Run
        instance = CustomConstraint(
            transform=dummy_transform,
            reverse_transform=dummy_reverse_transform,
            is_valid=is_valid_fqn
        )

        # Assert
        assert instance.transform == dummy_transform
        assert instance.reverse_transform == dummy_reverse_transform
        assert instance.is_valid == dummy_is_valid


class TestUniqueCombinations():

    def test___init__(self):
        """Test the ``UniqueCombinations.__init__`` method.

        It is expected to create a new Constraint instance and receiving the names of
        the columns that need to produce unique combinations.

        Side effects:
        - instance._colums == columns
        """
        # Setup
        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)

        # Assert
        assert instance._columns == columns

    def test_fit(self):
        """Test the ``UniqueCombinations.fit`` method.

        The ``UniqueCombinations.fit`` method is expected to:
        - Call ``UniqueCombinations._valid_separator``.
        - Find a valid separator for the data and generate the joint column name.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)

        # Run
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        instance.fit(table_data)

        # Asserts
        expected_combinations = pd.DataFrame({
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        assert instance._separator == '#'
        assert instance._joint_column == 'b#c'
        pd.testing.assert_frame_equal(instance._combinations, expected_combinations)

    def test_is_valid_true(self):
        """Test the ``UniqueCombinations.is_valid`` method.

        If the input data satisfies the constraint, result is a series of ``True`` values.

        Input:
        - Table data (pandas.DataFrame), satisfying the constraint.
        Output:
        - Series of ``True`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        out = instance.is_valid(table_data)

        expected_out = pd.Series([True, True, True], name='b#c')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false(self):
        """Test the ``UniqueCombinations.is_valid`` method.

        If the input data doesn't satisfy the constraint, result is a series of ``False`` values.

        Input:
        - Table data (pandas.DataFrame), which does not satisfy the constraint.
        Output:
        - Series of ``False`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        incorrect_table = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['D', 'E', 'F'],
            'c': ['g', 'h', 'i']
        })
        out = instance.is_valid(incorrect_table)

        # Assert
        expected_out = pd.Series([False, False, False], name='b#c')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_non_string_true(self):
        """Test the ``UniqueCombinations.is_valid`` method with non string columns.

        If the input data satisfies the constraint, result is a series of ``True`` values.

        Input:
        - Table data (pandas.DataFrame), satisfying the constraint.
        Output:
        - Series of ``True`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        out = instance.is_valid(table_data)

        expected_out = pd.Series([True, True, True], name='b#c#d')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_non_string_false(self):
        """Test the ``UniqueCombinations.is_valid`` method with non string columns.

        If the input data doesn't satisfy the constraint, result is a series of ``False`` values.

        Input:
        - Table data (pandas.DataFrame), which does not satisfy the constraint.
        Output:
        - Series of ``False`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        incorrect_table = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [6, 7, 8],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        out = instance.is_valid(incorrect_table)

        # Assert
        expected_out = pd.Series([False, False, False], name='b#c#d')
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform(self):
        """Test the ``UniqueCombinations.transform`` method.

        It is expected to return a Table data with the columns concatenated by the separator.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed, with the columns concatenated (pandas.DataFrame)
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        out = instance.transform(table_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out_a = pd.Series(['a', 'b', 'c'], name='a')
        pd.testing.assert_series_equal(expected_out_a, out['a'])
        try:
            [uuid.UUID(u) for c, u in out['b#c'].items()]
        except ValueError:
            assert False

    def test_transform_non_string(self):
        """Test the ``UniqueCombinations.transform`` method with non strings.

        It is expected to return a Table data with the columns concatenated by the separator.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed, with the columns as UUIDs.
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        out = instance.transform(table_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out_a = pd.Series(['a', 'b', 'c'], name='a')
        pd.testing.assert_series_equal(expected_out_a, out['a'])
        try:
            [uuid.UUID(u) for c, u in out['b#c#d'].items()]
        except ValueError:
            assert False

    def test_transform_not_all_columns_provided(self):
        """Test the ``UniqueCombinations.transform`` method.

        If some of the columns needed for the transform are missing, and
        ``fit_columns_model`` is False, it will raise a ``MissingConstraintColumnError``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Raises ``MissingConstraintColumnError``.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns, fit_columns_model=False)
        instance.fit(table_data)

        # Run/Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame({'a': ['a', 'b', 'c']}))

    def test_reverse_transform(self):
        """Test the ``UniqueCombinations.reverse_transform`` method.

        It is expected to return the original data separating the concatenated columns.

        Input:
        - Table data transformed (pandas.DataFrame)
        Output:
        - Original table data, with the concatenated columns separated (pandas.DataFrame)
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        transformed_data = instance.transform(table_data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_non_string(self):
        """Test the ``UniqueCombinations.reverse_transform`` method with a non string column.

        It is expected to return the original data separating the concatenated columns.

        Input:
        - Table data transformed (pandas.DataFrame)
        Output:
        - Original table data, with the concatenated columns separated (pandas.DataFrame)
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        transformed_data = instance.transform(table_data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        pd.testing.assert_frame_equal(expected_out, out)


class TestGreaterThan():

    def test___init___strict_false(self):
        """Test the ``GreaterThan.__init__`` method.

        The passed arguments should be stored as attributes.

        Input:
        - low = 'a'
        - high = 'b'
        Side effects:
        - instance._low == 'a'
        - instance._high == 'b'
        - instance._strict == False
        """
        # Run
        instance = GreaterThan(low='a', high='b')

        # Asserts
        assert instance._low == 'a'
        assert instance._high == 'b'
        assert instance._strict is False
        assert instance._high_is_scalar is None
        assert instance._low_is_scalar is None
        assert instance._drop is None

    def test___init___all_parameters_passed(self):
        """Test the ``GreaterThan.__init__`` method.

        The passed arguments should be stored as attributes.

        Input:
        - low = 'a'
        - high = 'b'
        - strict = True
        - drop = 'high'
        - high_is_scalar = True
        - low_is_scalar = False
        Side effects:
        - instance._low == 'a'
        - instance._high == 'b'
        - instance._stric == True
        - instance._drop = 'high'
        - instance._high_is_scalar = True
        - instance._low_is_scalar = False
        """
        # Run
        instance = GreaterThan(low='a', high='b', strict=True, drop='high',
                               high_is_scalar=True, low_is_scalar=False)

        # Asserts
        assert instance._low == 'a'
        assert instance._high == 'b'
        assert instance._strict is True
        assert instance._high_is_scalar is True
        assert instance._low_is_scalar is False
        assert instance._drop == 'high'

    def test_fit__low_is_scalar_is_none_determined_as_scalar(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should figure out if low is
        a scalar if ``_low_is_scalar`` is None.

        Input:
        - Table without ``low`` in columns.
        Side Effect:
        - ``_low_is_scalar`` should be set to ``True``.
        """
        # Setup
        instance = GreaterThan(low=3, high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._low_is_scalar is True

    def test_fit__low_is_scalar_is_none_determined_as_column(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should figure out if low is
        a column name if ``_low_is_scalar`` is None.

        Input:
        - Table with ``low`` in columns.
        Side Effect:
        - ``_low_is_scalar`` should be set to ``False``.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._low_is_scalar is False

    def test_fit__high_is_scalar_is_none_determined_as_scalar(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should figure out if high is
        a scalar if ``_high_is_scalar`` is None.

        Input:
        - Table without ``high`` in columns.
        Side Effect:
        - ``_high_is_scalar`` should be set to ``True``.
        """
        # Setup
        instance = GreaterThan(low='a', high=3)

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._high_is_scalar is True

    def test_fit__high_is_scalar_is_none_determined_as_column(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should figure out if high is
        a column name if ``_high_is_scalar`` is None.

        Input:
        - Table with ``high`` in columns.
        Side Effect:
        - ``_high_is_scalar`` should be set to ``False``.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._high_is_scalar is False

    def test_fit__high_is_scalar__low_is_scalar_raises_error(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should raise an error if
        `_low_is_scalar` and `_high_is_scalar` are true.

        Input:
        - Table with one column.
        Side Effect:
        - ``TypeError`` is raised.
        """
        # Setup
        instance = GreaterThan(low=1, high=2)

        # Run / Asserts
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(TypeError):
            instance.fit(table_data)

    def test_fit__column_to_reconstruct_drop_high(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should set ``_column_to_reconstruct``
        to ``instance._high`` if ``instance_drop`` is `high`.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_column_to_reconstruct`` is ``instance._high``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', drop='high')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._column_to_reconstruct == 'b'

    def test_fit__column_to_reconstruct_drop_low(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should set ``_column_to_reconstruct``
        to ``instance._low`` if ``instance_drop`` is `low`.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_column_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', drop='low')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._column_to_reconstruct == 'a'

    def test_fit__column_to_reconstruct_default(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should set ``_column_to_reconstruct``
        to `high` by default.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_column_to_reconstruct`` is ``instance._high``
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._column_to_reconstruct == 'b'

    def test_fit__column_to_reconstruct_high_is_scalar(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should set ``_column_to_reconstruct``
        to `low` if ``instance._high_is_scalar`` is ``True``.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_column_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._column_to_reconstruct == 'a'

    def test_fit__diff_column_one_column(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should set ``_diff_column``
        to the one column in ``instance.constraint_columns`` plus a
        token if there is only one column in that set.

        Input:
        - Table with one column.
        Side Effect:
        - ``_column_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high=3, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        instance.fit(table_data)

        # Asserts
        assert instance._diff_column == 'a#'

    def test_fit__diff_column_multiple_columns(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should set ``_diff_column``
        to the two columns in ``instance.constraint_columns`` separated
        by a token if there both columns are in that set.

        Input:
        - Table with two column.
        Side Effect:
        - ``_column_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._diff_column == 'a#b'

    def test_fit_int(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` and ``high_is_scalar`` are ``False``.

        Input:
        - Table that contains two constrained columns with the high one
          being made of integers.
        Side Effect:
        - The _dtype attribute gets `int` as the value even if the low
          column has a different dtype.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'i'

    def test_fit_float(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` and ``high_is_scalar`` are ``False``.

        Input:
        - Table that contains two constrained columns with the high one
          being made of float values.
        Side Effect:
        - The _dtype attribute gets `float` as the value even if the low
          column has a different dtype.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'f'

    def test_fit_datetime(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` and ``high_is_scalar`` are ``False``.

        Input:
        - Table that contains two constrained columns of datetimes.
        Side Effect:
        - The _dtype attribute gets `datetime` as the value.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01']),
            'b': pd.to_datetime(['2020-01-02'])
        })
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'M'

    def test_fit_type__high_is_scalar(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should learn and store the
        ``dtype`` of the ``low`` column as the ``_dtype`` attribute
        if ``_high_is_scalar`` is ``True``.

        Input:
        - Table that contains two constrained columns with the low one
          being made of floats.
        Side Effect:
        - The _dtype attribute gets `float` as the value.
        """
        # Setup
        instance = GreaterThan(low='a', high=3)

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'f'

    def test_fit_type__low_is_scalar(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` is ``True``.

        Input:
        - Table that contains two constrained columns with the high one
          being made of floats.
        Side Effect:
        - The _dtype attribute gets `float` as the value.
        """
        # Setup
        instance = GreaterThan(low=3, high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9]
        })
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'f'

    def test_is_valid_strict_false(self):
        """Test the ``GreaterThan.is_valid`` method with strict False.

        If strict is False, equal values should count as valid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - False should be returned for the strictly invalid row and True
          for the other two.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=False)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_strict_true(self):
        """Test the ``GreaterThan.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_low_is_scalar_high_is_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If low is a scalar, and high is a column name, then
        the values in that column should all be higher than
        ``instance._low``.

        Input:
        - Table with values above and below low.
        Output:
        - True should be returned for the rows where the high
        column is above low.
        """
        # Setup
        instance = GreaterThan(low=3, high='b', strict=False, low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False], name='b')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_high_is_scalar_low_is_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If high is a scalar, and low is a column name, then
        the values in that column should all be lower than
        ``instance._high``.

        Input:
        - Table with values above and below high.
        Output:
        - True should be returned for the rows where the low
        column is below high.
        """
        # Setup
        instance = GreaterThan(low='a', high=2, strict=False, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False], name='a')
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform_int_drop_none(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1.

        Setup:
        - ``_drop`` is set to ``None``, so all original columns will be in output.
        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._diff_column = 'a#b'

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_int_drop_high(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1. It should also drop the high column.

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4) and the high column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._diff_column = 'a#b'

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_int_drop_low(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1. It should also drop the low column.

        Setup:
        - ``_drop`` is set to ``low``.
        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4) and the low column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._diff_column = 'a#b'

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_float_drop_none(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type float.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1.

        Setup:
        - ``_drop`` is set to ``None``, so all original columns will be in output.
        Input:
        - Table with two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._diff_column = 'a#b'

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_datetime_drop_none(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type datetime.

        If the columns are of type datetime, ``transform`` is expected
        to convert the timedelta distance into numeric before applying
        the +1 and logarithm.

        Setup:
        - ``_drop`` is set to ``None``, so all original columns will be in output.
        Input:
        - Table with values at a distance of exactly 1 second.
        Output:
        - Same table with a diff column of the logarithms
          of the dinstance in nanoseconds + 1, which is np.log(1_000_000_001).
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._diff_column = 'a#b'
        instance._is_datetime = True

        # Run
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_not_all_columns_provided(self):
        """Test the ``GreaterThan.transform`` method.

        If some of the columns needed for the transform are missing, it will raise
        a ``MissingConstraintColumnError``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Raises ``MissingConstraintColumnError``.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, fit_columns_model=False)

        # Run/Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame({'a': ['a', 'b', 'c']}))

    def test_transform_high_is_scalar(self):
        """Test the ``GreaterThan.transform`` method with high as scalar.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high scalar value and the low column and create a diff column
        with the logarithm of the distance + 1.

        Setup:
        - ``_high`` is set to 5 and ``_high_is_scalar`` is ``True``.
        Input:
        - Table with one low column and two dummy columns.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low='a', high=5, strict=True, high_is_scalar=True)
        instance._diff_column = 'a#b'
        instance.constraint_columns = ['a']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(5), np.log(4), np.log(3)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_low_is_scalar(self):
        """Test the ``GreaterThan.transform`` method with high as scalar.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high scalar value and the low column and create a diff column
        with the logarithm of the distance + 1.

        Setup:
        - ``_high`` is set to 5 and ``_high_is_scalar`` is ``True``.
        Input:
        - Table with one low column and two dummy columns.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low=2, high='b', strict=True, low_is_scalar=True)
        instance._diff_column = 'a#b'
        instance.constraint_columns = ['b']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(3), np.log(4), np.log(5)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_int_drop_high(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to integers
            - add back the dropped column

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as int
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._diff_column = 'a#b'
        instance._column_to_reconstruct = 'b'

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_float_drop_high(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype float.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to float values
            - add back the dropped column

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as float values
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._dtype = np.dtype('float')
        instance._diff_column = 'a#b'
        instance._column_to_reconstruct = 'b'

        # Run
        transformed = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'c': [7, 8, 9],
            'b': [4.1, 5.2, 6.3],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_drop_high(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - add the low column
            - convert the output to datetimes

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with a diff column that contains the constant np.log(1_000_000_001).
        Output:
        - Same table with the high column replaced by the low one + one second
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._dtype = np.dtype('<M8[ns]')
        instance._diff_column = 'a#b'
        instance._is_datetime = True
        instance._column_to_reconstruct = 'b'

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01'])
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_int_drop_low(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high column
            - convert the output to integers
            - add back the dropped column

        Setup:
        - ``_drop`` is set to ``low``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the low column replaced by the high one - 3, as int
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._diff_column = 'a#b'
        instance._column_to_reconstruct = 'a'

        # Run
        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a': [1, 2, 3],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_drop_low(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - subtract from the high column
            - convert the output to datetimes

        Setup:
        - ``_drop`` is set to ``low``.
        Input:
        - Table with a diff column that contains the constant np.log(1_000_000_001).
        Output:
        - Same table with the low column replaced by the high one - one second
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._dtype = np.dtype('<M8[ns]')
        instance._diff_column = 'a#b'
        instance._is_datetime = True
        instance._column_to_reconstruct = 'a'

        # Run
        transformed = pd.DataFrame({
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00'])
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_int_drop_none(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        The table should have one invalid row where the low column is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low one + 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._diff_column = 'a#b'
        instance._column_to_reconstruct = 'b'

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 1, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_drop_none(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - add the low column when the row is invalid
            - convert the output to datetimes

        Setup:
        - ``_drop`` is set to ``None``.
        Input:
        - Table with a diff column that contains the constant np.log(1_000_000_001).
        The table should have one invalid row where the low column is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low one + one second
        for all invalid rows, and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = np.dtype('<M8[ns]')
        instance._diff_column = 'a#b'
        instance._is_datetime = True
        instance._column_to_reconstruct = 'b'

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-01T00:00:01']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2]
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_low_is_scalar(self):
        """Test the ``GreaterThan.reverse_transform`` method with low as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        - ``_low`` is set to an int and ``_low_is_scalar`` is ``True``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        The table should have one invalid row where the low value is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low value + 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low=3, high='b', strict=True, low_is_scalar=True)
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._diff_column = 'a#b'
        instance._column_to_reconstruct = 'b'

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 1, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 6, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_high_is_scalar(self):
        """Test the ``GreaterThan.reverse_transform`` method with high as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        - ``_high`` is set to an int and ``_high_is_scalar`` is ``True``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        The table should have one invalid row where the low column is
        higher than the high value.
        Output:
        - Same table with the low column replaced by the high one - 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high=3, strict=True, high_is_scalar=True)
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._diff_column = 'a#b'
        instance._column_to_reconstruct = 'a'

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 0],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)


class TestPositive():

    def test__init__(self):
        """
        Test the ``Positive.__init__`` method.

        The method is expected to set the ``_low`` instance variable
        to 0, the ``_low_is_scalar`` variable to ``True`` and the
        ``_high_is_scalar`` variable to ``False``. The rest of the
        parameters should be passed.

        Input:
        - strict = True
        - high = 'a'
        - drop = None
        Side effects:
        - instance._low == 0
        - instance._high == 'a'
        - instance._strict == True
        - instance._high_is_scalar = False
        - instance._low_is_scalar = True
        - instance._drop = None
        """
        # Run
        instance = Positive(high='a', strict=True, drop=None)

        # Asserts
        assert instance._low == 0
        assert instance._high == 'a'
        assert instance._strict is True
        assert instance._high_is_scalar is False
        assert instance._low_is_scalar is True
        assert instance._drop is None


class TestNegative():

    def test__init__(self):
        """
        Test the ``Negative.__init__`` method.

        The method is expected to set the ``_high`` instance variable
        to 0, the ``_high_is_scalar`` variable to ``True`` and the
        ``_low_is_scalar`` variable to ``False``. The rest of the
        parameters should be passed.

        Input:
        - strict = True
        - low = 'a'
        - drop = None
        Side effects:
        - instance._low == 'a'
        - instance._high == 0
        - instance._strict == True
        - instance._high_is_scalar = True
        - instance._low_is_scalar = False
        - instance._drop = None
        """
        # Run
        instance = Negative(low='a', strict=True, drop=None)

        # Asserts
        assert instance._low == 'a'
        assert instance._high == 0
        assert instance._strict is True
        assert instance._high_is_scalar is True
        assert instance._low_is_scalar is False
        assert instance._drop is None


def new_column(data):
    """Formula to be used for the ``TestColumnFormula`` class."""
    return data['a'] + data['b']


class TestColumnFormula():

    def test___init__(self):
        """Test the ``ColumnFormula.__init__`` method.

        It is expected to create a new Constraint instance
        and import the formula to use for the computation.

        Input:
        - column = 'c'
        - formula = new_column
        """
        # Setup
        column = 'c'

        # Run
        instance = ColumnFormula(column=column, formula=new_column)

        # Assert
        assert instance._column == column
        assert instance._formula == new_column

    def test_is_valid_valid(self):
        """Test the ``ColumnFormula.is_valid`` method for a valid data.

        If the data fulfills the formula, result is a series of ``True`` values.

        Input:
        - Table data fulfilling the formula (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_non_valid(self):
        """Test the ``ColumnFormula.is_valid`` method for a non-valid data.

        If the data does not fulfill the formula, result is a series of ``False`` values.

        Input:
        - Table data not fulfilling the formula (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        })
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform(self):
        """Test the ``ColumnFormula.transform`` method.

        It is expected to drop the indicated column from the table.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data without the indicated column (pandas.DataFrame)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform(self):
        """Test the ``ColumnFormula.reverse_transform`` method.

        It is expected to compute the indicated column by applying the given formula.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 1, 1]
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        pd.testing.assert_frame_equal(expected_out, out)


class TestRounding():

    def test___init__(self):
        """Test the ``Rounding.__init__`` method.

        It is expected to create a new Constraint instance
        and set the rounding args.

        Input:
        - columns = ['b', 'c']
        - digits = 2
        """
        # Setup
        columns = ['b', 'c']
        digits = 2

        # Run
        instance = Rounding(columns=columns, digits=digits)

        # Assert
        assert instance._columns == columns
        assert instance._digits == digits

    def test___init__invalid_digits(self):
        """Test the ``Rounding.__init__`` method with an invalid argument.

        Pass in an invalid ``digits`` argument, and expect a ValueError.

        Input:
        - columns = ['b', 'c']
        - digits = 20
        """
        # Setup
        columns = ['b', 'c']
        digits = 20

        # Run
        with pytest.raises(ValueError):
            Rounding(columns=columns, digits=digits)

    def test___init__invalid_tolerance(self):
        """Test the ``Rounding.__init__`` method with an invalid argument.

        Pass in an invalid ``tolerance`` argument, and expect a ValueError.

        Input:
        - columns = ['b', 'c']
        - digits = 2
        - tolerance = 0.1
        """
        # Setup
        columns = ['b', 'c']
        digits = 2
        tolerance = 0.1

        # Run
        with pytest.raises(ValueError):
            Rounding(columns=columns, digits=digits, tolerance=tolerance)

    def test_is_valid_positive_digits(self):
        """Test the ``Rounding.is_valid`` method for a positive digits argument.

        Input:
        - Table data with desired decimal places (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        columns = ['b', 'c']
        digits = 2
        tolerance = 1e-3
        instance = Rounding(columns=columns, digits=digits, tolerance=tolerance)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [4.12, 5.51, None, 6.941, 1.129],
            'c': [5.315, 7.12, 1.12, 9.131, 12.329],
            'd': ['a', 'b', 'd', 'e', None],
            'e': [123.31598, -1.12001, 1.12453, 8.12129, 1.32923]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, True, False, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_negative_digits(self):
        """Test the ``Rounding.is_valid`` method for a negative digits argument.

        Input:
        - Table data with desired decimal places (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        columns = ['b']
        digits = -2
        tolerance = 1
        instance = Rounding(columns=columns, digits=digits, tolerance=tolerance)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [401, 500, 6921, 799, None],
            'c': [5.3134, 7.1212, 9.1209, 101.1234, None],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_zero_digits(self):
        """Test the ``Rounding.is_valid`` method for a zero digits argument.

        Input:
        - Table data not with the desired decimal places (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        columns = ['b', 'c']
        digits = 0
        tolerance = 1e-4
        instance = Rounding(columns=columns, digits=digits, tolerance=tolerance)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, None, 3, 4],
            'b': [4, 5.5, 1.2, 6.0001, 5.99999],
            'c': [5, 7.12, 1.31, 9.00001, 4.9999],
            'd': ['a', 'b', None, 'd', 'e'],
            'e': [2.1254, 17.12123, 124.12, 123.0112, -9.129434]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_reverse_transform_positive_digits(self):
        """Test the ``Rounding.reverse_transform`` method with positive digits.

        Expect that the columns are rounded to the specified integer digit.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        digits = 3
        instance = Rounding(columns=columns, digits=digits)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, None, 4],
            'b': [4.12345, None, 5.100, 6.0001, 1.7999],
            'c': [1.1, 1.234, 9.13459, 4.3248, 6.1312],
            'd': ['a', 'b', 'd', 'e', None]
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3, None, 4],
            'b': [4.123, None, 5.100, 6.000, 1.800],
            'c': [1.100, 1.234, 9.135, 4.325, 6.131],
            'd': ['a', 'b', 'd', 'e', None]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_negative_digits(self):
        """Test the ``Rounding.reverse_transform`` method with negative digits.

        Expect that the columns are rounded to the specified integer digit.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        columns = ['b']
        digits = -3
        instance = Rounding(columns=columns, digits=digits)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [41234.5, None, 5000, 6001, 5928],
            'c': [1.1, 1.23423, 9.13459, 12.12125, 18.12152],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [41000.0, None, 5000.0, 6000.0, 6000.0],
            'c': [1.1, 1.23423, 9.13459, 12.12125, 18.12152],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_zero_digits(self):
        """Test the ``Rounding.reverse_transform`` method with zero digits.

        Expect that the columns are rounded to the specified integer digit.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        digits = 0
        instance = Rounding(columns=columns, digits=digits)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [4.12345, None, 5.0, 6.01, 7.9],
            'c': [1.1, 1.0, 9.13459, None, 8.89],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [4.0, None, 5.0, 6.0, 8.0],
            'c': [1.0, 1.0, 9.0, None, 9.0],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        pd.testing.assert_frame_equal(expected_out, out)


def transform(data, low, high):
    """Transform to be used for the TestBetween class."""
    data = (data - low) / (high - low) * 0.95 + 0.025
    return np.log(data / (1.0 - data))


class TestBetween():

    def test_transform_scalar_scalar(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [4, 5, 6],
        })
        instance.fit(table_data)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'a#0.0#1.0': transform(table_data[column], low, high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_transform_scalar_column(self):
        """Test the ``Between.transform`` method with ``low`` as scalar and ``high`` as a column.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 6],
        })
        instance.fit(table_data)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0.5, 1, 6],
            'a#0.0#b': transform(table_data[column], low, table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_transform_column_scalar(self):
        """Test the ``Between.transform`` method with ``low`` as a column and ``high`` as scalar.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
        })
        instance.fit(table_data)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0, -1, 0.5],
            'a#b#1.0': transform(table_data[column], table_data[low], high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_transform_column_column(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as columns.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6]
        })
        instance.fit(table_data)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            'a#b#c': transform(table_data[column], table_data[low], table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_scalar_scalar(self):
        """Test ``Between.reverse_transform`` with ``low`` and ``high`` as scalars.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True,
                           low_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [4, 5, 6],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            'a#0.0#1.0': transform(table_data[column], low, high)
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_scalar_column(self):
        """Test ``Between.reverse_transform`` with ``low`` as scalar and ``high`` as a column.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [0.5, 1, 6],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [0.5, 1, 6],
            'a#0.0#b': transform(table_data[column], low, table_data[high])
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_scalar(self):
        """Test ``Between.reverse_transform`` with ``low`` as a column and ``high`` as scalar.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [0, -1, 0.5],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [0, -1, 0.5],
            'a#b#1.0': transform(table_data[column], table_data[low], high)
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_column(self):
        """Test ``Between.reverse_transform`` with ``low`` and ``high`` as columns.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        table_data = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            'a#b#c': transform(table_data[column], table_data[low], table_data[high])
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_is_valid_strict_true(self):
        """Test the ``Between.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a valid row, a strictly invalid row and an
          invalid row. (pandas.DataFrame)
        Output:
        - True should be returned for the valid row and False
          for the other two. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, strict=True, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 1, 3],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out, check_names=False)

    def test_is_valid_strict_false(self):
        """Test the ``Between.is_valid`` method with strict False.

        If strict is False, equal values should count as valid.

        Input:
        - Table with a valid row, a strictly invalid row and an
          invalid row. (pandas.DataFrame)
        Output:
        - True should be returned for the first two rows, and False
          for the last one (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, strict=False, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 1, 3],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out, check_names=False)

    def test_is_valid_scalar_column(self):
        """Test the ``Between.is_valid`` method with ``low`` as scalar and ``high`` as a column.

        Is expected to return whether the constraint ``column`` is between the
        ``low`` and ``high`` values.

        Input:
        - Table data where the last value is greater than ``high``. (pandas.DataFrame)
        Output:
        - True should be returned for the two first rows, False
          for the last one. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 0.6],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_column_scalar(self):
        """Test the ``Between.is_valid`` method with ``low`` as a column and ``high`` as scalar.

        Is expected to return whether the constraint ``column`` is between the
        ``low`` and ``high`` values.

        Input:
        - Table data where the second value is smaller than ``low`` and
          last value is greater than ``high``. (pandas.DataFrame)
        Output:
        - True should be returned for the first row, False
          for the last two. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 1.9],
            'b': [-0.5, 1, 0.6],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_column_column(self):
        """Test the ``Between.is_valid`` method with ``low`` and ``high`` as columns.

        Is expected to return whether the constraint ``column`` is between the
        ``low`` and ``high`` values.

        Input:
        - Table data where the last value is greater than ``high``. (pandas.DataFrame)
        Output:
        - True should be returned for the two first rows, False
          for the last one. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 0.6]
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])

        pd.testing.assert_series_equal(expected_out, out)


class TestOneHotEncoding():

    def test_reverse_transform(self):
        """Test the ``OneHotEncoding.reverse_transform`` method.

        It is expected to, for each of the appropriate rows, set the column
        with the largest value to one and set all other columns to zero.

        Input:
        - Table data with any numbers (pandas.DataFrame)
        Output:
        - Table data where the appropriate rows are one hot (pandas.DataFrame)
        """
        # Setup
        instance = OneHotEncoding(columns=['a', 'b'])

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.8],
            'b': [0.8, 0.1, 0.9],
            'c': [1, 2, 3]
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [0.0, 1.0, 0.0],
            'b': [1.0, 0.0, 1.0],
            'c': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_is_valid(self):
        """Test the ``OneHotEncoding.is_valid`` method.

        ``True`` when for the rows where the data is one hot, ``False`` otherwise.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` and ``False`` values (pandas.Series)
        """
        # Setup
        instance = OneHotEncoding(columns=['a', 'b', 'c'])

        # Run
        table_data = pd.DataFrame({
            'a': [1.0, 1.0, 0.0, 1.0],
            'b': [0.0, 1.0, 0.0, 0.5],
            'c': [0.0, 2.0, 0.0, 0.0],
            'd': [1, 2, 3, 4]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test__sample_constraint_columns_proper(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to return a table with the appropriate complementary column ``b``,
        since column ``a`` is entirely defined by the ``condition`` table.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where ``a`` is the same as in ``condition``
          and ``b`` is complementary`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
        })
        instance = OneHotEncoding(columns=['a', 'b'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0, 0.0, 0.0] * 5,
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.0, 0.0, 0.0] * 5,
            'b': [0.0, 1.0, 1.0] * 5,
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_one_one(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Since the condition column contains a one for all rows, expected to assign
        all other columns to zeros.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where the first column contains one's and others columns zero's (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [0.0] * 10,
            'c': [0.0] * 10
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_two_ones(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since the condition contains two ones
        in a single row.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [1.0] * 10,
            'c': [0.0] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test__sample_constraint_columns_non_binary(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since the condition contains a non binary value.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.5] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test__sample_constraint_columns_all_zeros(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since the condition contains only zeros.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 0] * 5,
            'b': [0, 1] * 5,
            'c': [0, 0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0] * 10,
            'b': [0.0] * 10,
            'c': [0.0] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test__sample_constraint_columns_valid_condition(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to generate a table where every column satisfies the ``condition``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table satifying the ``condition`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0] * 10,
            'b': [1.0] * 10,
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'a': [0.0] * 10,
            'b': [1.0] * 10,
            'c': [0.0] * 10
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_one_zero(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Since the condition column contains only one zero, expected to randomly sample
        from unset columns any valid possibility. Since the ``b`` column in ``data``
        contains all the ones, it's expected to return a table where only ``b`` has ones.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where ``b`` is all one`s and other columns are all zero`s (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [0.0, 0.0] * 5,
            'b': [1.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'c': [0.0] * 10,
            'a': [0.0] * 10,
            'b': [1.0] * 10
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_one_zero_alt(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Since the condition column contains only one zero, expected to randomly sample
        from unset columns any valid possibility.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where ``c`` is all zero`s and ``b`` xor ``a`` is always one (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        assert (out['c'] == 0.0).all()
        assert ((out['b'] == 1.0) ^ (out['a'] == 1.0)).all()

    def test_sample_constraint_columns_list_of_conditions(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to generate a table satisfying the ``condition``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table satisfying the ``condition`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_output = pd.DataFrame({
            'a': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5,
            'b': [1.0, 0.0] * 5
        })
        pd.testing.assert_frame_equal(out, expected_output)

    def test_sample_constraint_columns_negative_values(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since condition is not a one hot vector.
        This tests that even if the sum of a row is one it still crashes.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [-1.0] * 10,
            'c': [1.0] * 10
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [-1.0] * 10,
            'c': [1.0] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test_sample_constraint_columns_all_zeros_but_one(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to generate a table where column ``a`` is filled with ones,
        and ``b`` and ``c`` filled with zeros.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table satisfying the ``condition`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0] * 10,
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_output = pd.DataFrame({
            'a': [0.0] * 10,
            'c': [0.0] * 10,
            'b': [1.0] * 10
        })
        pd.testing.assert_frame_equal(out, expected_output)
