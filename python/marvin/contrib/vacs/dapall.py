# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 15:13:07
# @Last modified by: José Sánchez-Gallego
# @Last Modified time: 2018-07-08 13:28:26

from __future__ import absolute_import, division, print_function

import astropy
from marvin.core.exceptions import MarvinError

from .base import VACMixIn


class DapVAC(VACMixIn):
    """Provides access to the DAPall VAC.

    VAC name: dapall

    URL: https://www.sdss.org/dr15/manga/manga-data/catalogs/#DAPALLFile

    Description: The DAPall file contains a per-galaxy summary of the DAP
    output. This file is intended only as an example / template for other VACs,
    and it is not included in Marvin as a VAC. DAPall data can be access in
    Marvin using the .dapall attribute.

    """

    name = 'dapall'

    def get_data(self, parent_object):

        path_params = {'drpver': parent_object._drpver,
                       'dapver': parent_object._dapver}

        if not self.file_exists(path_params=path_params):
            filename = self.download_vac(path_params=path_params)
        else:
            filename = self.get_path(path_params=path_params)

        dap_table = astropy.table.Table.read(filename)

        plateifu = parent_object.plateifu

        rows = dap_table[dap_table['PLATEIFU'] == plateifu]

        if len(rows) == 0:
            raise MarvinError('cannot match plate-ifu with VAC data.')

        if hasattr(parent_object, 'bintype') and hasattr(parent_object, 'template'):
            dap_type = (str(parent_object.bintype).upper() + '-' +
                        str(parent_object.template).upper())
            row = rows[rows['DAPTYPE'] == dap_type]
        else:
            return rows

        if len(rows) == 0:
            raise MarvinError('cannot match plate-ifu with VAC data.')
        elif len(row) == 1:
            return row[0]
        else:
            raise MarvinError('unexpected error matching plate-ifu to VAC data: '
                              'multiple rows found.')
