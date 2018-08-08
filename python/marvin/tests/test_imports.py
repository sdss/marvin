#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-09
# @Filename: test_imports.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-07-09 12:11:48

import marvin


class TestImports(object):

    def test_access_tools_full_path(self):

        assert marvin.tools.plate.Plate is not None
        assert marvin.tools.cube.Cube is not None
        assert marvin.tools.maps.Maps is not None
        assert marvin.tools.modelcube.ModelCube is not None
        assert marvin.tools.spaxel.Spaxel is not None
        assert marvin.tools.spaxel.Bin is not None
        assert marvin.tools.image.Image is not None

    def test_access_tools_from_root(self):

        assert marvin.tools.Plate is not None
        assert marvin.tools.Cube is not None
        assert marvin.tools.Maps is not None
        assert marvin.tools.ModelCube is not None
        assert marvin.tools.Spaxel is not None
        assert marvin.tools.Bin is not None
        assert marvin.tools.Image is not None
