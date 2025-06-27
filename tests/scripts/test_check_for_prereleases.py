from scripts.check_for_prereleases import get_dev_dependencies


def test_get_dev_dependencies():
    """Test get_dev_dependencies ignores regular releases."""
    # Setup
    dependencies = ['rdt>=1.1.1', 'sdv>=1.0.2']

    # Run
    dev_dependencies = get_dev_dependencies(dependency_list=dependencies)

    # Assert
    assert len(dev_dependencies) == 0


def test_get_dev_dependencies_prereleases():
    """Test get_dev_dependencies detects prereleases."""
    # Setup
    dependencies = ['rdt>=1.1.1.dev0', 'sdv>=1.0.2.rc1']

    # Run
    dev_dependencies = get_dev_dependencies(dependency_list=dependencies)

    # Assert
    assert dev_dependencies == dependencies


def test_get_dev_dependencies_url():
    """Test get_dev_dependencies detects url requirements."""
    # Setup
    dependencies = ['rdt>=1.1.1', 'sdv @ git+https://github.com/sdv-dev/sdv.git@main']

    # Run
    dev_dependencies = get_dev_dependencies(dependency_list=dependencies)

    # Assert
    assert dev_dependencies == ['sdv @ git+https://github.com/sdv-dev/sdv.git@main']
