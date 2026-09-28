import sdv


def test_sdv_versions():
    """Test version for SDV."""
    assert sdv.version.__all__ == ('community', 'enterprise')
    assert sdv.version.community == sdv.__version__
    assert sdv.version.enterprise is None
