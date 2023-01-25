from unittest.mock import Mock, patch

from sdv.data_processing.utils import load_module_from_path


@patch('sdv.data_processing.utils.importlib')
def test_load_module_from_path(mock_importlib):
    """Test the ``load_module_from_path``.

    Test that given a ``PosixPath`` this loads the module and returns it.
    """
    # Setup
    path = Mock()
    path.exists.return_value = True
    path.parent = Mock()
    path.parent.name = 'example'
    path.name = 'myfile.py'

    # Run
    result = load_module_from_path(path)

    # Assert
    spec = mock_importlib.util.spec_from_file_location.return_value
    module = mock_importlib.util.module_from_spec.return_value
    mock_importlib.util.spec_from_file_location.assert_called_once_with('example.myfile', path)
    mock_importlib.util.module_from_spec.assert_called_once_with(spec)
    spec.loader.exec_module.assert_called_once_with(module)

    assert result == module
