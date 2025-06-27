"""Tests for the ``tasks.py`` file."""

from tasks import _get_minimum_versions


def test_get_minimum_versions():
    """Test the ``_get_minimum_versions`` method.

    The method should return the minimum versions of the dependencies for the given python version.
    If a library is linked to an URL, the minimum version should be the URL.
    """
    # Setup
    dependencies = [
        "numpy>=1.20.0,<2;python_version<'3.10'",
        "numpy>=1.23.3,<2;python_version>='3.10'",
        "pandas>=1.2.0,<2;python_version<'3.10'",
        "pandas>=1.3.0,<2;python_version>='3.10'",
        'humanfriendly>=8.2,<11',
        'pandas @ git+https://github.com/pandas-dev/pandas.git@master',
    ]

    # Run
    minimum_versions_39 = _get_minimum_versions(dependencies, '3.9')
    minimum_versions_310 = _get_minimum_versions(dependencies, '3.10')

    # Assert
    expected_versions_39 = [
        'numpy==1.20.0',
        'git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'humanfriendly==8.2',
    ]
    expected_versions_310 = [
        'numpy==1.23.3',
        'git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'humanfriendly==8.2',
    ]

    assert minimum_versions_39 == expected_versions_39
    assert minimum_versions_310 == expected_versions_310
