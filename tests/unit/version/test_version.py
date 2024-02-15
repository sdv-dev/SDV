import sdv


def test_sdv_versions():
    """Test version for SDV."""
    assert sdv.version.__all__ == ('public', 'enterprise')
    assert sdv.version.public == sdv.__version__
    assert sdv.version.enterprise is None
