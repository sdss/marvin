import os
from marvin.utils.plot import colorbar


def test_linear_lab_filename_exists():
    assert os.path.isfile(colorbar.linear_Lab_filename())
