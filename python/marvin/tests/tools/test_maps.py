#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-06-22
# @Filename: test_maps.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-06 12:12:16


from __future__ import absolute_import, division, print_function

import copy
import re
from os.path import join

import astropy
import astropy.io.fits
import numpy as np
import pytest

import marvin
import marvin.tools.spaxel
from marvin.core.exceptions import MarvinError
from marvin.tests import marvin_test_if
from marvin.tools.maps import Maps
from marvin.utils.datamodel.dap.base import Property

from ..conftest import set_the_config


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

    @pytest.mark.parametrize('objtype, errmsg',
                             [('cube', 'Trying to open a non DAP file with Marvin Maps'),
                              ('models', 'Trying to open a DAP LOGCUBE with Marvin Maps')])
    def test_maps_wrong_file(self, galaxy, objtype, errmsg):
        path = galaxy.cubepath if objtype == 'cube' else galaxy.modelpath
        with pytest.raises(MarvinError) as cm:
            Maps(filename=path)
        assert errmsg in str(cm.value)

    def test_nobintype_in_db(self, galaxy):

        if galaxy.release != 'MPL-6':
            pytest.skip('only running this test for MPL6')

        with pytest.raises(MarvinError) as cm:
            maps = Maps(plateifu=galaxy.plateifu, bintype='ALL', release=galaxy.release)

        assert 'Specified bintype ALL is not available in the DB' in str(cm.value)

    @pytest.mark.slow
    def test_datamodel(self, galaxy, exporigin):

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))

        fin = 'manga-{0}-{1}-MAPS-{2}.fits.gz'.format(galaxy.plate, galaxy.ifu, galaxy.bintemp)
        path = join(galaxy.mangaanalysis, galaxy.drpver, galaxy.dapver, galaxy.bintemp,
                    str(galaxy.plate), galaxy.ifu, fin)
        hdus = astropy.io.fits.open(path)

        for hdu in hdus[1:]:

            if ('IVAR' in hdu.name) or ('MASK' in hdu.name) or ('SIGMACORR' in hdu.name):
                continue

            name = hdu.name.lower()
            data = hdu.data
            header = hdu.header

            if len(data.shape) < 3:
                val = maps.getMap(name, exact=True).value
                assert val == pytest.approx(data, 0.0001), name

            else:
                for kk, vv in header.items():
                    channel_num = re.match('^[0-9]+$', kk[1:])

                    if (kk[0] == 'C') and (channel_num is not None):
                        channel = '_'.join(re.findall(r"[\w']+", vv)).lower()
                        fullname = '_'.join((name, channel))
                        val = maps.getMap(fullname, exact=True).value
                        assert val == pytest.approx(data[int(kk[1:]) - 1], 0.0001), name

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
        assert redshift == pytest.approx(galaxy.redshift)

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

    def test_getMapRatio(self, galaxy):
        maps = Maps(galaxy.plateifu)
        map_ratio = maps.getMapRatio('emline_gflux', 'nii_6585', 'ha_6564')
        map_arith = maps.emline_gflux_nii_6585 / maps.emline_gflux_ha_6564

        assert map_ratio.value == pytest.approx(map_arith.value, nan_ok=True)
        assert map_ratio.ivar == pytest.approx(map_arith.ivar, nan_ok=True)
        assert map_ratio.mask == pytest.approx(map_arith.mask, nan_ok=True)


class TestMaskbit(object):

    @marvin_test_if(mark='include', maps_release_only=dict(release=['MPL-4']))
    def test_quality_flag_mpl4(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.quality_flag is None

    @pytest.mark.parametrize('flag',
                             ['manga_target1',
                              'manga_target2',
                              'manga_target3',
                              'target_flags'])
    def test_flag(self, flag, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert getattr(ha, flag, None) is not None
