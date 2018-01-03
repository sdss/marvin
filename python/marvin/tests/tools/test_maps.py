#!/usr/bin/env python
# encoding: utf-8
#
# test_maps.py
#
# Created by José Sánchez-Gallego on 22 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy

import pytest
import numpy as np
import astropy
import astropy.io.fits

import marvin
from marvin.tools.maps import Maps
import marvin.tools.spaxel
from marvin.core.exceptions import MarvinError
from marvin.utils.datamodel.dap.base import Property
from marvin.tests import marvin_test_if


def _assert_maps(maps, galaxy):
    """Basic checks for a Maps object."""

    assert maps is not None
    assert maps.plateifu == galaxy.plateifu
    assert maps.mangaid == galaxy.mangaid
    assert maps.wcs is not None
    assert maps.bintype == galaxy.bintype

    assert len(maps._shape) == len(galaxy.shape)
    for ii in range(len(maps._shape)):
        assert maps._shape[ii] == galaxy.shape[ii]


class TestMaps(object):

    def _get_maps_kwargs(self, galaxy, data_origin):

        if data_origin == 'file':
            maps_kwargs = dict(filename=galaxy.mapspath)
        else:
            maps_kwargs = dict(plateifu=galaxy.plateifu, release=galaxy.release,
                               bintype=galaxy.bintype, template=galaxy.template,
                               mode='local' if data_origin == 'db' else 'remote')

        return maps_kwargs

    def test_load(self, galaxy, exporigin):

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))

        _assert_maps(maps, galaxy)

        assert maps.data_origin == exporigin

        if exporigin == 'file':
            assert isinstance(maps.data, astropy.io.fits.HDUList)
        elif exporigin == 'db':
            assert isinstance(maps.data, marvin.marvindb.dapdb.File)

        cube = maps.getCube()

        assert cube is not None
        assert cube.plateifu == galaxy.plateifu
        assert cube.mangaid == galaxy.mangaid

    @pytest.mark.parametrize('monkeyconfig', [('release', 'MPL-5')], indirect=True)
    def test_load_mpl4_global_mpl5(self, galaxy, monkeyconfig, data_origin):

        assert marvin.config.release == 'MPL-5'
        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))

        assert maps.release == galaxy.release
        assert maps._drpver == galaxy.drpver
        assert maps._dapver == galaxy.dapver

    def test_maps_redshift(self, maps, galaxy):
        redshift = maps.nsa.redshift \
            if maps.release == 'MPL-4' and maps.data_origin == 'file' else maps.nsa.z
        assert pytest.approx(redshift, galaxy.redshift)

    def test_release(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        assert maps.release == galaxy.release

    def test_set_release_fails(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        with pytest.raises(MarvinError) as ee:
            maps.release = 'a'
            assert 'the release cannot be changed' in str(ee.exception)

    def test_deepcopy(self, galaxy):
        maps1 = Maps(plateifu=galaxy.plateifu)
        maps2 = copy.deepcopy(maps1)

        for attr in vars(maps1):
            if not attr.startswith('_'):
                value = getattr(maps1, attr)
                value2 = getattr(maps2, attr)

                if isinstance(value, np.ndarray):
                    assert np.isclose(value, value2).all()

                elif isinstance(value, astropy.wcs.wcs.WCS):
                    for key in vars(value):
                        assert getattr(value, key) == getattr(value2, key)

                elif isinstance(value, marvin.tools.cube.Cube):
                    pass

                elif (isinstance(value, list) and len(value) > 0 and
                      isinstance(value[0], Property)):
                        for property1, property2 in zip(value, value2):
                            assert property1 == property2

                else:
                    assert value == value2, attr


class TestMaskbit(object):

    @marvin_test_if(mark='include', maps_release_only=dict(release=['MPL-4']))
    def test_quality_flag_mpl4(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.quality_flag is None

    @marvin_test_if(mark='skip', maps_release_only=dict(release=['MPL-4']))
    def test_quality_flag(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.quality_flag.mask == 0

    @pytest.mark.parametrize('flag',
                             ['manga_target1',
                              'manga_target2',
                              'manga_target3',
                              'target_flags'])
    def test_flag(self, flag, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert getattr(ha, flag, None) is not None
