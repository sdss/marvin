# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 18:57:07
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-21 19:07:00

from __future__ import print_function, division, absolute_import
from marvin.contrib.vacs.base import VACMixIn
from sdss_access.path import Path


class TestVAC(VACMixIn):

    def _get_from_file(self):
        self._filename = Path().full('dapall', dapver='2.1.3', drpver='v2_3_1')

    @property
    def test_vac_row(self):
        return {'testvac': 'this is a test'}

