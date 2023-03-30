from unittest.mock import Mock, patch

from sdv import _add_version


@patch('sdv.iter_entry_points')
def test__add_version(entry_points_mock):
    # Setup
    entry_point = Mock()
    entry_points_mock.return_value = [entry_point]

    # Run
    _add_version()

    # Assert
    entry_points_mock.assert_called_once_with(name='version', group='sdv_modules')


@patch('sdv.warnings.warn')
@patch('sdv.iter_entry_points')
def test__add_version_bad_addon(entry_points_mock, warning_mock):
    # Setup
    def entry_point_error():
        raise ValueError()

    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.module = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    msg = 'Failed to load "bad_entry_point" from "bad_module".'

    # Run
    _add_version()

    # Assert
    entry_points_mock.assert_called_once_with(name='version', group='sdv_modules')
    warning_mock.assert_called_once_with(msg)
