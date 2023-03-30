from unittest.mock import Mock, patch

from sdv._addons import _find_addons


@patch('sdv._addons.iter_entry_points')
def test__add_version(entry_points_mock):
    # Setup
    entry_point = Mock()
    entry_point.name = 'entry_name'
    entry_point.load.return_value = 'entry_point'
    entry_points_mock.return_value = [entry_point]
    test_dict = {}

    # Run
    _find_addons(group='group', parent_globals=test_dict)

    # Assert
    entry_points_mock.assert_called_once_with(group='group')
    assert test_dict['entry_name'] == 'entry_point'


@patch('sdv._addons.warnings.warn')
@patch('sdv._addons.iter_entry_points')
def test__add_version_bad_addon(entry_points_mock, warning_mock):
    # Setup
    def entry_point_error():
        raise ValueError()

    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.module = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    test_dict = {}
    msg = 'Failed to load "bad_entry_point" from "bad_module".'

    # Run
    _find_addons(group='group', parent_globals=test_dict)

    # Assert
    entry_points_mock.assert_called_once_with(group='group')
    warning_mock.assert_called_once_with(msg)
