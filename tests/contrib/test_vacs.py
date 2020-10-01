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

import astropy.io.fits
import pytest

from marvin.contrib.vacs import VACMixIn
from marvin.tools.maps import Maps


class TestVACs(object):

    def test_subclasses(self):

        assert len(VACMixIn.__subclasses__()) > 0

    def test_mangahi(self):

        my_map = Maps('7443-12701')

        assert hasattr(my_map, 'vacs')
        #assert my_map.vacs.mangahi is not None  # figure out how to test based on release

    def test_vac_container(self):

        my_map = Maps('8485-1901')

        assert my_map.vacs.__class__.__name__ == 'VACContainer'
        assert list(my_map.vacs) is not None

    def test_vacs_return(self, plateifu, release):

        if release in ['MPL-4', 'MPL-5', 'MPL-6', 'MPL-8']:
            pytest.skip()

        vacs = VACMixIn.__subclasses__()

        for vac in vacs:
            for include_class in vac.include:
                __ = importlib.import_module(str(include_class.__module__))
                obj = include_class(plateifu, release=release)

                assert hasattr(obj, 'vacs')
                assert hasattr(obj.vacs, vac.name)
                assert getattr(obj.vacs, vac.name) is not None


@pytest.mark.xfail(reason="will not work with tested releases it does not have")
class TestMangaHI(object):

    def test_return_type(self, plateifu):

        my_map = Maps(plateifu)
        assert isinstance(my_map.vacs.HI, object)
