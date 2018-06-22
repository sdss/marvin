# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 15:13:07
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-21 19:08:03

from __future__ import print_function, division, absolute_import

from marvin.contrib.vacs.base import VACMixIn
from sdss_access.path import Path


class DapVAC(VACMixIn):

    def _get_from_file(self):
        self._filename = Path().full('dapall', dapver='2.1.3', drpver='v2_3_1')

    @property
    def dap_vac_row(self):
        return {'ddapvac': 'dfdf', 'stuff': self.plateifu}


