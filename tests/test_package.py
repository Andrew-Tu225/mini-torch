import minitorch


def test_package_exposes_version():
    assert minitorch.__version__ == "0.1.0"
