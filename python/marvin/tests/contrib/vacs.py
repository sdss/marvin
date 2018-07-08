#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-08
# @Filename: vacs.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego
# @Last modified time: 2018-07-08 14:10:48

import marvin
from marvin.contrib.vacs import VACMixIn
from marvin.contrib.vacs.dapall import DapVAC
from marvin.tools.maps import Maps


class TestVACs(object):

    def test_subclasses(self):

        assert len(VACMixIn.__subclasses__()) > 0

    def test_galaxyzoo3d(self):

        my_map = Maps('8485-1901')

        assert hasattr(my_map, 'vacs')
        assert my_map.vacs.galaxyzoo3d is not None

    def test_file_exists(self):

        dapall_vac = DapVAC()

        drpver, dapver = marvin.config.lookUpVersions()
        assert dapall_vac.file_exists(path_params={'drpver': drpver, 'dapver': dapver})
