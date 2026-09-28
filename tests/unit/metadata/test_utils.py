import pytest

from sdv.metadata.utils import (
    _format_column_metadata,
    _format_metadata_value,
    _print_primary_key_detection,
)


@pytest.mark.parametrize(
    'value,expected',
    [
        (True, 'True'),
        (False, 'False'),
        (None, 'None'),
        ('id', "'id'"),
        ('categorical', "'categorical'"),
        ('', "''"),
    ],
)
def test__format_metadata_value(value, expected):
    """Test ``_format_metadata_value`` formats bools/None with no quotes and strings with quotes."""
    # Run
    result = _format_metadata_value(value)

    # Assert
    assert result == expected


def test__format_column_metadata_sdtype_only():
    """Test ``_format_column_metadata`` formats a dict with only sdtype."""
    # Setup
    sdtype_info = {'sdtype': 'categorical'}

    # Run
    result = _format_column_metadata(sdtype_info)

    # Assert
    assert result == "sdtype='categorical'"


def test__format_column_metadata_with_kwargs():
    """Test ``_format_column_metadata`` formats a dict with sdtype and additional kwargs."""
    # Setup
    sdtype_info = {'sdtype': 'numerical', 'computer_representation': 'Float'}

    # Run
    result = _format_column_metadata(sdtype_info)

    # Assert
    assert result == "sdtype='numerical', computer_representation='Float'"


def test__format_column_metadata_sdtype_reordered_to_front():
    """Test ``_format_column_metadata`` moves sdtype to the front regardless of insertion order."""
    # Setup
    sdtype_info = {
        'datetime_format': '%Y-%m-%d',
        'pii': False,
        'sdtype': 'datetime',
    }

    # Run
    result = _format_column_metadata(sdtype_info)

    # Assert
    assert result == "sdtype='datetime', datetime_format='%Y-%m-%d', pii=False"


def test__format_column_metadata_mixed_value_types():
    """Test ``_format_column_metadata`` quotes strings and leaves bools/None unquoted."""
    # Setup
    sdtype_info = {
        'sdtype': 'datetime',
        'datetime_format': None,
        'pii': True,
    }

    # Run
    result = _format_column_metadata(sdtype_info)

    # Assert
    assert result == "sdtype='datetime', datetime_format=None, pii=True"


@pytest.mark.parametrize(
    'sdtype_updated,pii_removed,expected',
    [
        (False, False, "- primary_key='user_id'\n"),
        (True, False, "- primary_key='user_id' (updating sdtype to 'id')\n"),
        (False, True, "- primary_key='user_id' (removing 'pii' field)\n"),
        (
            True,
            True,
            "- primary_key='user_id' (updating sdtype to 'id', removing 'pii' field)\n",
        ),
    ],
)
def test__print_primary_key_detection(capsys, sdtype_updated, pii_removed, expected):
    """Test ``_print_primary_key_detection`` prints the PK with and without notes."""
    # Run
    _print_primary_key_detection('user_id', sdtype_updated, pii_removed)

    # Assert
    assert capsys.readouterr().out == expected


def test__print_primary_key_detection_no_pk(capsys):
    """Test ``_print_primary_key_detection`` prints a fallback message when no PK ."""
    # Run
    _print_primary_key_detection(None, False, False)

    # Assert
    assert capsys.readouterr().out == '- No primary key found\n'
