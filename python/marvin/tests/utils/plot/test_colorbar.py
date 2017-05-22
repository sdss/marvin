import os
from marvin.utils.plot import colorbar


def test_linearlab_filename_exists():
    assert os.path.isfile(colorbar._linearlab_filename())
