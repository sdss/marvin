#!/usr/bin/env python
# encoding: utf-8
#
# test_maps.py
#
# Created by José Sánchez-Gallego on 22 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest

import astropy.io.fits

import marvin
from marvin.tools.maps import Maps
import marvin.tools.spaxel
from marvin.core.exceptions import MarvinError
from marvin.tests import marvin_test_if


def _assert_maps(maps, galaxy):
    """Basic checks for a Maps object."""

    assert maps is not None
    assert maps.plateifu == galaxy.plateifu
    assert maps.mangaid == galaxy.mangaid
    assert maps.wcs is not None
    assert maps.bintype == galaxy.bintype

    assert len(maps.shape) == len(galaxy.shape)
    for ii in range(len(maps.shape)):
        assert maps.shape[ii] == galaxy.shape[ii]


class TestMaps(object):

    def _get_maps_kwargs(self, galaxy, data_origin):

        if data_origin == 'file':
            maps_kwargs = dict(filename=galaxy.mapspath)
        else:
            maps_kwargs = dict(plateifu=galaxy.plateifu, release=galaxy.release,
                               bintype=galaxy.bintype, template_kin=galaxy.template,
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

        assert maps.cube is not None
        assert maps.cube.plateifu == galaxy.plateifu
        assert maps.cube.mangaid == galaxy.mangaid

    @pytest.mark.parametrize('monkeyconfig', [('release', 'MPL-5')], indirect=True)
    def test_load_mpl4_global_mpl5(self, galaxy, monkeyconfig, data_origin):

        assert marvin.config.release == 'MPL-5'
        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))

        assert maps._release == galaxy.release
        assert maps._drpver == galaxy.drpver
        assert maps._dapver == galaxy.dapver

    def test_get_spaxel(self, galaxy, data_origin):

        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')

        assert isinstance(spaxel, marvin.tools.spaxel.Spaxel)
        assert spaxel.spectrum is not None
        assert len(spaxel.properties.keys()) > 0

        expected = galaxy.stellar_vel_ivar_x15_y8_lower[galaxy.release][galaxy.template]
        assert spaxel.properties['stellar_vel'].ivar == pytest.approx(expected, abs=1e-6)

    def test_get_spaxel_test2(self, galaxy, data_origin):

        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))
        spaxel = maps.getSpaxel(x=5, y=5)

        assert isinstance(spaxel, marvin.tools.spaxel.Spaxel)
        assert spaxel.spectrum is not None
        assert len(spaxel.properties.keys()) > 0

    def test_get_spaxel_no_db(self, galaxy, exporigin):
        """Tests getting an spaxel if there is no DB."""

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))
        spaxel = maps.getSpaxel(x=5, y=5)

        assert spaxel.maps.data_origin == exporigin

        assert isinstance(spaxel, marvin.tools.spaxel.Spaxel)
        assert spaxel.spectrum is not None
        assert len(spaxel.properties.keys()) > 0

    def test_maps_redshift(self, maps, galaxy):
        redshift = maps.nsa.redshift if maps.release == 'MPL-4' and maps.data_origin == 'file' else maps.nsa.z
        assert pytest.approx(redshift, galaxy.redshift)

    def test_release(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        assert maps.release == galaxy.release

    def test_set_release_fails(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        with pytest.raises(MarvinError) as ee:
            maps.release = 'a'
            assert 'the release cannot be changed' in str(ee.exception)
