# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: test_vacs.py
# Project: tools
# Author: Brian Cherinka
# Created: Sunday, 15th September 2019 7:56:52 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2019 Brian Cherinka
# Last Modified: Sunday, 15th September 2019 8:15:11 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import
import pytest
from marvin import config
from marvin.tools.vacs import VACs


@pytest.fixture(scope='session')
def vacs():
    config.setRelease('DR17')
    v = VACs()
    yield v
    v = None


class TestVacs(object):

    def test_listvacs(self, vacs):
        assert len(vacs.list_vacs()) > 0

    def test_check_target(self, vacs):
        tdict = vacs.check_target('1-209232')
        assert tdict == {'firefly': {'miles': True, 'mastar': True},
                         'galaxyzoo': {'GZD_auto': False, 'gzUKIDSS': False, 'gz': True},
                         'gz3d': True, 'gema': True, 'HI': False}

    def test_check_target_fail(self, vacs):
        with pytest.raises(AssertionError,
                           match='Identifier "plateifu" is not available.  Try a "mangaid'):
            vacs.check_target('8485-1901')


class TestVACDataClass(object):

    def test_gzvac(self, vacs):
        gz = vacs.galaxyzoo
        'gz' in vacs.galaxyzoo
        gg = gz['gz']
        assert gg.name == 'galaxyzoo'
        assert gg._path is not None
        assert gg.data is not None

    def test_has_target(self, vacs):
        gema = vacs.gema
        assert gema.has_target('1-209232') == True

    def test_addmethod(self, vacs):
        hi = vacs.HI
        assert hasattr(hi, 'plot_mass_fraction')

