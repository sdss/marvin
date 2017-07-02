import os

import pytest

from marvin.utils.plot import colorbar


def test_linearlab_filename_exists():
    assert os.path.isfile(colorbar._linearlab_filename())

@pytest.mark.parametrize('cbrange, expected',
                         [([1.5, 1.75], []),
                          ([1.5, 4], [2, 3]),
                          ([1.5, 8], [2, 3, 6]),
                          ([1.5, 18], [2, 3, 6, 10]),
                          ([1.5, 45], [2, 3, 6, 10, 20, 30]),
                          ([2, 8], [2, 3, 6]),
                          ([3, 25], [3, 6, 10, 20])
                          ])
def test_log_cbticks(cbrange, expected):
    assert (colorbar._log_cbticks(cbrange) == expected).all()
    assert (colorbar._set_cbticks(cbrange, {'log_cb': True})[1] == expected).all()

