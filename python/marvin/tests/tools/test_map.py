#!/usr/bin/env python
# encoding: utf-8
#
# test_map.py
#
# Created by Brett Andrews on 2 Jul 2017.

import numpy as np
import astropy
import matplotlib
import pytest

from marvin.tools.maps import Maps
from marvin.tools.map import Map
from marvin.tests import marvin_test_if


def _get_maps_kwargs(galaxy, data_origin):

    if data_origin == 'file':
        maps_kwargs = dict(filename=galaxy.mapspath)
    else:
        maps_kwargs = dict(plateifu=galaxy.plateifu, release=galaxy.release,
                           bintype=galaxy.bintype, template_kin=galaxy.template,
                           mode='local' if data_origin == 'db' else 'remote')

    return maps_kwargs


@pytest.fixture(scope='function', params=[('emline_gflux', 'ha_6564'),
                                          ('emline_gvel', 'oiii_5008'),
                                          ('stellar_vel', None),
                                          ('stellar_sigma', None)])
def map_(request, galaxy, data_origin):

    maps = Maps(**_get_maps_kwargs(galaxy, data_origin))
    map_ = maps.getMap(property_name=request.param[0], channel=request.param[1])
    map_.data_origin = data_origin
    return map_


class TestMap(object):

    def test_map(self, map_, galaxy):
        assert map_.release == galaxy.release

        assert tuple(map_.shape) == tuple(galaxy.shape)
        assert map_.value.shape == tuple(galaxy.shape)
        assert map_.ivar.shape == tuple(galaxy.shape)
        assert map_.mask.shape == tuple(galaxy.shape)

        assert (map_.masked.data == map_.value).all()
        assert (map_.masked.mask == map_.mask.astype(bool)).all()

        assert pytest.approx(map_.snr, np.abs(map_.value * np.sqrt(map_.ivar)))

        assert map_.header['BUNIT'] == map_.unit

        assert isinstance(map_.header, astropy.io.fits.header.Header)

    def test_plot(self, map_):
        fig, ax = map_.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)

    @marvin_test_if(map_={'data_origin': ['db']}, mark='skip')
    def test_save_and_restore(self, temp_scratch, map_):

        fout = temp_scratch.join('test_map.mpf')
        map_.save(str(fout))
        assert fout.check() is True

        map_restored = Map.restore(str(fout), delete=True)
        assert tuple(map_.shape) == tuple(map_restored.shape)
