from unittest.mock import Mock, patch

from sdv._addons import _find_addons


@patch('sdv._addons.iter_entry_points')
def test__find_addons(entry_points_mock):
    """Test loading an add-on."""
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


@patch('sdv._addons.iter_entry_points')
def test__find_addons_add_all(entry_points_mock):
    """Test loading an add-on and all submodules."""
    # Setup
    entry_point = Mock()
    module = Mock()
    module.__all__ = ['addon1', 'addon2', 'duplicate']
    module.addon1 = 'addon1_object'
    module.addon2 = 'addon2_object'
    module.duplicate = 'addon3_object'
    entry_point.name = 'entry_name'
    entry_point.load.return_value = module
    entry_points_mock.return_value = [entry_point]
    test_dict = {
        'duplicate': 'original_duplicate',
        '__all__': ('duplicate',)
    }

    # Run
    _find_addons(group='group', parent_globals=test_dict, add_all=True)

    # Assert
    entry_points_mock.assert_called_once_with(group='group')
    assert test_dict['entry_name'] == module
    assert test_dict['addon1'] == module.addon1
    assert test_dict['addon2'] == module.addon2
    assert test_dict['duplicate'] == 'original_duplicate'
    assert set(test_dict['__all__']) == {'duplicate', 'addon1', 'addon2'}


@patch('sdv._addons.iter_entry_points')
def test__find_addons_add_all_no___all___attr(entry_points_mock):
    """Test loading an add-on without an __all__ attribute."""
    # Setup
    entry_point = Mock()
    module = Mock()
    entry_point.name = 'entry_name'
    entry_point.load.return_value = module
    entry_points_mock.return_value = [entry_point]
    test_dict = {}

    # Run
    _find_addons(group='group', parent_globals=test_dict, add_all=True)

    # Assert
    entry_points_mock.assert_called_once_with(group='group')
    assert test_dict['entry_name'] == module


@patch('sdv._addons.warnings.warn')
@patch('sdv._addons.iter_entry_points')
def test__find_addons_bad_addon(entry_points_mock, warning_mock):
    """Test failing to load an add-on generates a warning."""
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
