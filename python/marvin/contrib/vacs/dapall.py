# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 15:13:07
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-22 12:12:08

from __future__ import print_function, division, absolute_import

from marvin.contrib.vacs.base import VACMixIn
from sdss_access.path import Path
from astropy.io import fits
from astropy.table import Table


class DapVAC(VACMixIn):

    @property
    def dap_vac_row(self):
        ''' Return a single row from the DAPall file for a given plateifu '''

        # get the full path
        vac_filename = Path().full('dapall', dapver=self._dapver, drpver=self._drpver)

        # open the DAPall file
        hdu = fits.open(vac_filename)
        dapall_table = Table(hdu[1].data)

        daptype = self.bintype.name + '-' + self.template.name

        dapall_row = dapall_table[(dapall_table['PLATEIFU'] == self.plateifu) &
                                  (dapall_table['DAPTYPE'] == daptype)]

        return dapall_row


