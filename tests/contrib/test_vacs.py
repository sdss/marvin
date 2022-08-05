#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-08
# @Filename: vacs.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-07-09 17:27:59

import importlib

import pytest

from marvin.contrib.vacs import VACMixIn
from marvin.contrib.vacs.hi import HITarget
from marvin.tools.maps import Maps


class TestVACs(object):

    def test_subclasses(self):

        assert len(VACMixIn.__subclasses__()) > 0

    def test_mangahi(self):

        my_map = Maps('7443-12701')

        assert hasattr(my_map, 'vacs')
        assert my_map.vacs.HI is not None  # figure out how to test based on release

    def test_vac_container(self):

        my_map = Maps('8485-1901')

        assert my_map.vacs.__class__.__name__ == 'VACContainer'
        assert list(my_map.vacs) is not None

    def test_vacs_return(self, plateifu, release):

        if release not in ['DR17']:
            pytest.skip()

        for vac in ['HI', 'gema', 'galaxyzoo']:
            obj = Maps(plateifu, release=release)
            assert hasattr(obj, 'vacs')
            assert vac in obj.vacs
            assert obj.vacs[vac] is not None


class TestMangaHI(object):

    def test_return_type(self, plateifu):

        my_map = Maps(plateifu)
        assert isinstance(my_map.vacs.HI, HITarget)
