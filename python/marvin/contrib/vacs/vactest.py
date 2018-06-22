# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 18:57:07
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-22 14:00:02

from __future__ import print_function, division, absolute_import
from marvin.contrib.vacs.base import VACMixIn


class TestVAC(VACMixIn):

    @property
    def test_vac_row(self):
        return {'testvac': 'this is a test'}

