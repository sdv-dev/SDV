"""Methods to help with upgrading metadata."""

import warnings

from sdv.constraints import (
    FixedCombinations, Inequality, Negative, OneHotEncoding, Positive, Range, ScalarInequality,
    ScalarRange, Unique)


def _upgrade_columns_and_keys(old_metadata):
    new_metadata = {}
    columns = {}
    fields = old_metadata.get('fields')
    alternate_keys = []
    primary_key = old_metadata.get('primary_key')
    for field, field_meta in fields.items():
        column_meta = {}
        old_type = field_meta['type']
        subtype = field_meta.get('subtype')
        column_meta['sdtype'] = old_type

        if old_type == 'numerical':
            if subtype == 'float':
                column_meta['computer_representation'] = 'Float'
            elif subtype == 'integer':
                column_meta['computer_representation'] = 'Int64'

        elif old_type == 'datetime':
            datetime_format = field_meta.get('format')
            if datetime_format:
                column_meta['datetime_format'] = datetime_format

        elif old_type == 'id':
            column_meta['sdtype'] = 'id'
            if subtype == 'integer':
                regex_format = r'\d{30}'
            else:
                regex_format = field_meta.get('regex', '[A-Za-z]{5}')

            if not field_meta.get('pii'):
                column_meta['regex_format'] = regex_format

            if field != primary_key and field_meta.get('ref') is None:
                alternate_keys.append(field)

        if field_meta.get('pii'):
            column_meta['pii'] = True
            pii_category = field_meta.get('pii_category')
            if isinstance(pii_category, list):
                column_meta['sdtype'] = pii_category[0]
            elif pii_category:
                column_meta['sdtype'] = pii_category

        columns[field] = column_meta

    new_metadata['columns'] = columns
    new_metadata['primary_key'] = primary_key
    if alternate_keys:
        new_metadata['alternate_keys'] = alternate_keys

    return new_metadata


def _upgrade_positive_negative(old_constraint):
    new_constraints = []
    strict = old_constraint.get('strict')
    old_name = old_constraint.get('constraint')
    is_positive = old_name == 'sdv.constraints.tabular.Positive'
    constraint_name = Positive.__name__ if is_positive else Negative.__name__
    columns = old_constraint.get('columns')
    if not isinstance(columns, list):
        columns = [columns]

    for column in columns:
        new_constraint = {
            'constraint_name': constraint_name,
            'column_name': column,
            'strict_boundaries': strict
        }
        new_constraints.append(new_constraint)

    return new_constraints


def _upgrade_unique_combinations(old_constraint):
    new_constraint = {
        'constraint_name': FixedCombinations.__name__,
        'column_names': old_constraint.get('columns')
    }

    return [new_constraint]


def _upgrade_greater_than(old_constraint):
    scalar = old_constraint.get('scalar')
    high = old_constraint.get('high')
    low = old_constraint.get('low')
    high_is_string = isinstance(high, str)
    low_is_string = isinstance(low, str)
    strict = old_constraint.get('strict', False)
    new_constraints = []

    if scalar is None:
        high_has_multiple = isinstance(high, list) and len(high) != 1
        low_has_multiple = isinstance(low, list) and len(low) != 1
        high_column_name = f"'{high}'" if high_is_string else high
        low_column_name = f"'{low}'" if low_is_string else low
        if high_has_multiple or low_has_multiple:
            warnings.warn(
                f"Unable to upgrade the GreaterThan constraint specified for 'high' "
                f"{high_column_name} and 'low' {low_column_name}. Manually add "
                f'{Inequality.__name__} constraints to capture this logic.'
            )
            return []

        new_constraint = {
            'constraint_name': Inequality.__name__,
            'high_column_name': high if high_is_string else high[0],
            'low_column_name': low if low_is_string else low[0],
            'strict_boundaries': strict
        }
        new_constraints.append(new_constraint)

    elif scalar == 'low':
        high = [high] if high_is_string else high
        for column in high:
            new_constraint = {
                'constraint_name': ScalarInequality.__name__,
                'column_name': column,
                'relation': '>' if strict else '>=',
                'value': low
            }
            new_constraints.append(new_constraint)

    else:
        low = [low] if low_is_string else low
        for column in low:
            new_constraint = {
                'constraint_name': ScalarInequality.__name__,
                'column_name': column,
                'relation': '<' if strict else '<=',
                'value': high
            }
            new_constraints.append(new_constraint)

    return new_constraints


def _upgrade_between(old_constraint):
    high_is_scalar = old_constraint.get('high_is_scalar')
    low_is_scalar = old_constraint.get('low_is_scalar')
    high = old_constraint.get('high')
    low = old_constraint.get('low')
    constraint_column = old_constraint.get('constraint_column')
    strict = old_constraint.get('strict', False)
    new_constraints = []
    if high_is_scalar and low_is_scalar:
        new_constraint = {
            'constraint_name': ScalarRange.__name__,
            'column_name': constraint_column,
            'low_value': low,
            'high_value': high,
            'strict_boundaries': strict
        }
        new_constraints.append(new_constraint)

    elif high_is_scalar and not low_is_scalar:
        inequality_constraint = {
            'constraint_name': Inequality.__name__,
            'low_column_name': low,
            'high_column_name': constraint_column,
            'strict_boundaries': strict
        }
        scalar_constraint = {
            'constraint_name': ScalarInequality.__name__,
            'column_name': constraint_column,
            'relation': '<' if strict else '<=',
            'value': high
        }
        new_constraints.append(inequality_constraint)
        new_constraints.append(scalar_constraint)

    elif not high_is_scalar and low_is_scalar:
        inequality_constraint = {
            'constraint_name': Inequality.__name__,
            'low_column_name': constraint_column,
            'high_column_name': high,
            'strict_boundaries': strict
        }
        scalar_constraint = {
            'constraint_name': ScalarInequality.__name__,
            'column_name': constraint_column,
            'relation': '>' if strict else '>=',
            'value': low
        }
        new_constraints.append(inequality_constraint)
        new_constraints.append(scalar_constraint)

    else:
        new_constraint = {
            'constraint_name': Range.__name__,
            'low_column_name': low,
            'middle_column_name': constraint_column,
            'high_column_name': high,
            'strict_boundaries': strict
        }
        new_constraints.append(new_constraint)

    return new_constraints


def _upgrade_one_hot_encoding(old_constraint):
    new_constraint = {
        'constraint_name': OneHotEncoding.__name__,
        'column_names': old_constraint.get('columns')
    }
    return [new_constraint]


def _upgrade_unique(old_constraint):
    new_constraint = {
        'constraint_name': Unique.__name__,
        'column_names': old_constraint.get('columns')
    }
    return [new_constraint]


def _upgrade_constraint(old_constraint):
    new_constraints = []
    constraint_name = old_constraint.get('constraint')
    if constraint_name in ('sdv.constraints.tabular.Positive', 'sdv.constraints.tabular.Negative'):
        new_constraints = _upgrade_positive_negative(old_constraint)

    elif constraint_name == 'sdv.constraints.tabular.UniqueCombinations':
        new_constraints = _upgrade_unique_combinations(old_constraint)

    elif constraint_name == 'sdv.constraints.tabular.GreaterThan':
        new_constraints = _upgrade_greater_than(old_constraint)

    elif constraint_name == 'sdv.constraints.tabular.Between':
        new_constraints = _upgrade_between(old_constraint)

    elif constraint_name == 'sdv.constraints.tabular.Rounding':
        warnings.warn(
            "The SDV no longer supports the 'Rounding' constraint. Most models automatically "
            'round to the same number of decimal digits as the original data.'
        )

    elif constraint_name == 'sdv.constraints.tabular.OneHotEncoding':
        new_constraints = _upgrade_one_hot_encoding(old_constraint)

    elif constraint_name == 'sdv.constraints.tabular.Unique':
        new_constraints = _upgrade_unique(old_constraint)

    elif constraint_name == 'sdv.constraints.tabular.ColumnFormula':
        warnings.warn(
            'This method does not upgrade ColumnFormula constraints. Please convert your '
            'logic using the new CustomConstraint API.'
        )

    elif constraint_name == 'sdv.constraints.tabular.CustomConstraint':
        warnings.warn(
            'This method does not upgrade CustomConstraint objects. Please convert '
            'your logic using the new CustomConstraint API.'
        )

    else:
        warnings.warn(
            f'Unable to upgrade the {constraint_name} constraint. Please add in the constraint '
            'using the new Constraints API.'
        )

    return new_constraints


def _upgrade_constraints(old_metadata):
    old_constraints = old_metadata.get('constraints')
    if not old_constraints:
        return None

    needs_upgrading = 'constraint' in old_constraints[0].keys()
    if not needs_upgrading:
        return old_constraints

    new_constraints = []
    for constraint in old_constraints:
        new_constraint = _upgrade_constraint(constraint)
        new_constraints += new_constraint

    return new_constraints


def convert_metadata(old_metadata):
    """Convert old metadata to the new metadata format.

    Args:
        old_metadata (dict):
            A dict version of the old metadata.

    Returns:
        An equivalent dict in the new metadata format.
    """
    new_metadata = _upgrade_columns_and_keys(old_metadata)
    return new_metadata
