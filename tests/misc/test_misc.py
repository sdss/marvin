# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-12 18:20:03
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-07-05 14:22:58

from __future__ import print_function, division, absolute_import

import pytest
import os

from marvin import config
from marvin.tools.cube import Cube


class TestMisc(object):

    @pytest.mark.parametrize('mpl, drpver', [('DR17', 'v3_1_1')])
    def test_custom_drpall(self, galaxy, mpl, drpver):
        assert galaxy.drpall in config.drpall
        cube = Cube(plateifu=galaxy.plateifu, release=mpl)
        drpall = 'drpall-{0}.fits'.format(drpver)
        assert cube._release == mpl
        assert cube._drpver == drpver
        assert os.path.exists(cube._drpall) is True
        assert drpall in cube._drpall
        assert galaxy.drpall in config.drpall
