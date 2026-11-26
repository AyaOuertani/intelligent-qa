import qa


def test_package_is_importable_and_versioned():
    assert qa.__version__ == "0.1.0"
