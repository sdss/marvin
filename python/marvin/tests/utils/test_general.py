#!/usr/bin/env python
# encoding: utf-8
"""

test_general.py

Created by José Sánchez-Gallego on 7 Apr 2016.
Licensed under a 3-clause BSD license.

Revision history:
    7 Apr 2016 J. Sánchez-Gallego
      Initial version
    15 Jun 2016 B. Andrews
      Converted to pytest

"""

from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import marvin
from marvin.tools.maps import Maps
from marvin.tools.quantities import Map
from marvin.tools.cube import Cube
from marvin.tools.quantities import Spectrum
from marvin.utils.general.structs import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.utils.general import (convertCoords, get_nsa_data, getWCSFromPng, get_plot_params,
                                  _sort_dir, getDapRedux, getDefaultMapPath)
from marvin.utils.datamodel.dap.plotting import get_default_plot_params


@pytest.fixture(scope='function')
def wcs(galaxy):
    return WCS(fits.getheader(galaxy.cubepath, 1))


class TestConvertCoords(object):

    @pytest.mark.parametrize('pifu, expected',
                             [('7443-12701', [[36, 36],
                                              [39, 41],
                                              [37, 31],
                                              [31, 37],
                                              [46, 46],
                                              [26, 26],
                                              [38, 38],
                                              [36, 36]]),
                              ('8485-1901', [[17, 17],
                                             [20, 22],
                                             [18, 12],
                                             [12, 18],
                                             [27, 27],
                                             [7, 7],
                                             [20, 18],
                                             [17, 17]])])
    def test_pix_center(self, galaxy, pifu, expected):
        """Tests mode='pix', xyorig='center'."""

        if galaxy.plateifu != pifu:
            pytest.skip('Skipping non-matching plateifu.')

        coords = [[0, 0],
                  [5, 3],
                  [-5, 1],
                  [1, -5],
                  [10, 10],
                  [-10, -10],
                  [1.5, 2.5],
                  [0.4, 0.25]]

        cubeCoords = convertCoords(coords, mode='pix', shape=galaxy.shape)
        assert cubeCoords == pytest.approx(np.array(expected))

    def test_pix_lower(self, galaxy):
        """Tests mode='pix', xyorig='lower'."""

        coords = [[0, 0],
                  [5, 3],
                  [10, 10],
                  [1.5, 2.5],
                  [0.4, 0.25]]

        expected = [[0, 0],
                    [3, 5],
                    [10, 10],
                    [2, 2],
                    [0, 0]]

        cubeCoords = convertCoords(coords, mode='pix', shape=galaxy.shape,
                                   xyorig='lower')
        assert cubeCoords == pytest.approx(np.array(expected))

    @pytest.mark.parametrize('naxis0, coords',
                             [(72, [[230.51104, 43.531993],
                                    [230.50912, 43.530743],
                                    [230.50797, 43.534215],
                                    [230.50932, 43.534215]]),
                              (34, [[232.5447, 48.690201],
                                    [232.54259, 48.688948],
                                    [232.54135, 48.692415],
                                    [232.54285, 48.692372]])])
    def test_sky(self, wcs, naxis0, coords):
        """Tests mode='sky'."""

        print('wcs', wcs._naxis[0])
        if wcs._naxis[0] != naxis0:
            pytest.skip('Skipping non-matching cube size.')

        expected = [[17, 17],
                    [8, 27],
                    [33, 33],
                    [33, 26]]

        cubeCoords = convertCoords(np.array(coords), mode='sky', wcs=wcs)

        assert cubeCoords == pytest.approx(np.array(expected))

    @pytest.mark.parametrize('coords, mode, xyorig',
                             [([-100, 0], 'pix', 'center'),
                              ([100, 100], 'pix', 'center'),
                              ([-100, 0], 'pix', 'lower'),
                              ([100, 100], 'pix', 'lower'),
                              ([230, 48], 'sky', None),
                              ([233, 48], 'sky', None)],
                             ids=['-50_0_cen', '50_50_cen', '-50_0_low', '50_50_low',
                                  '230_48_sky', '233_48_sky'])
    def test_coords_outside_cube(self, coords, mode, xyorig, galaxy, wcs):
        kwargs = {'coords': coords, 'mode': mode}

        if xyorig is not None:
            kwargs['xyorig'] = xyorig

        if kwargs['mode'] == 'sky':
            kwargs['wcs'] = wcs

        with pytest.raises(MarvinError) as cm:
            convertCoords(shape=galaxy.shape, **kwargs)

        assert 'some indices are out of limits' in str(cm.value)


class TestGetNSAData(object):

    def _test_nsa(self, galaxy, data):
        assert isinstance(data, DotableCaseInsensitive)
        assert 'iauname' in data.keys()
        assert data['iauname'] == galaxy.iauname
        assert data['iauname'] == data.iauname
        assert 'profmean_ivar' in data.keys()
        assert 'elpetro_mag_g' in data
        assert 'version' not in data.keys()
        assert data['elpetro_mag_g'] == data.elpetro_mag_g

    def _test_drpall(self, galaxy, data):
        assert isinstance(data, DotableCaseInsensitive)
        assert 'version' in data.keys()
        assert 'profmean_ivar' not in data.keys()
        if galaxy.release != 'MPL-4':
            assert 'iauname' in data.keys()
            assert data['iauname'] == galaxy.iauname
            assert data['iauname'] == data.iauname
            assert 'elpetro_absmag' in data.keys()

    @pytest.mark.parametrize('source', [('nsa'), ('drpall')])
    def test_nsa(self, galaxy, mode, db, source):
        if mode == 'local' and source == 'nsa' and marvin.config.db is None:
            with pytest.raises(MarvinError) as cm:
                data = get_nsa_data(galaxy.mangaid, source=source, mode=mode)
            errmsg = 'get_nsa_data: cannot find a valid DB connection.'
            assert cm.type == MarvinError
            assert errmsg in str(cm.value)
        else:
            data = get_nsa_data(galaxy.mangaid, source=source, mode=mode)
            if source == 'nsa':
                self._test_nsa(galaxy, data)
            elif source == 'drpall':
                self._test_drpall(galaxy, data)

    @pytest.mark.parametrize('monkeyconfig', [('_drpall', None)], indirect=True, ids=['nodrpall'])
    def test_nodrpall(self, galaxy, monkeyconfig):
        data = get_nsa_data(galaxy.mangaid, source='drpall', mode='auto')
        self._test_drpall(galaxy, data)


class TestPillowImage(object):

    def test_image_has_wcs(self, galaxy):
        w = getWCSFromPng(galaxy.imgpath)
        assert isinstance(w, WCS) is True

    def test_use_pil(self):
        try:
            import PIL
        except ImportError as e:
            with pytest.raises(ImportError) as cm:
                err = 'No module named PIL'
                assert err == str(e.args[0])


class TestDataModelPlotParams(object):

    @pytest.mark.parametrize('name, desired',
                             [('emline_gflux', {'cmap': 'linearlab', 'percentile_clip': [5, 95], 'symmetric': False, 'snr_min': 1}),
                              ('stellar_vel', {'cmap': 'RdBu_r', 'percentile_clip': [10, 90], 'symmetric': True, 'snr_min': None}),
                              ('stellar_sigma', {'cmap': 'inferno', 'percentile_clip': [10, 90], 'symmetric': False, 'snr_min': 1})],
                             ids=['emline', 'stvel', 'stsig'])
    def test_get_plot_params(self, dapver, name, desired):
        params = get_default_plot_params(dapver)

        if 'vel' in name:
            key = 'vel'
        elif 'sigma' in name:
            key = 'sigma'
        else:
            key = 'default'

        desired['bitmasks'] = params[key]['bitmasks']
        actual = get_plot_params(dapver=dapver, prop=name)
        assert desired == actual


class TestSortDir(object):

    @pytest.mark.parametrize('class_, expected',
                             [(Map, ['error', 'inst_sigma_correction', 'ivar',
                                     'getMaps', 'mask', 'masked', 'plot',
                                     'restore', 'save', 'snr', 'value', 'from_maps',
                                     'binid', 'descale', 'datamodel', 'pixmask',
                                     'quality_flag', 'target_flags', 'manga_target1',
                                     'manga_target2', 'manga_target3'])])
    def test_sort_dir_map(self, galaxy, class_, expected):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']

        dir_ = _sort_dir(ha, class_)
        dir_public = [it for it in dir_ if it[0] is not '_']
        assert set(dir_public) == set(expected)

    @pytest.mark.parametrize('class_, expected',
                             [(Spectrum, ['error', 'masked', 'plot', 'snr', 'ivar', 'mask',
                                          'wavelength', 'value', 'descale'])])
    def test_sort_dir_spectrum(self, galaxy, class_, expected):
        cube = Cube(plateifu=galaxy.plateifu)
        spax = cube[0, 0]
        spec = spax.flux

        dir_ = _sort_dir(spec, class_)
        dir_public = [it for it in dir_ if it[0] is not '_']
        assert set(dir_public) == set(expected)


class TestGetDapRedux(object):

    def test_success(self, release, versions):
        path = getDapRedux(release)
        verpath = '/'.join(versions)

        base = 'https://data.sdss.org/sas/mangawork/manga/spectro/analysis'
        full = '{0}/{1}'.format(base, verpath)
        assert 'default' not in path
        assert base in path
        assert path == full


class TestGetDefaultMapPath(object):

    def test_success(self, galaxy):
        path = getDefaultMapPath(release=galaxy.release, plate=galaxy.plate, ifu=galaxy.ifu,
                                 daptype=galaxy.bintemp, mode='MAPS')
        verpath = '/'.join((galaxy.drpver, galaxy.dapver))
        base = 'https://data.sdss.org/sas/mangawork/manga/spectro/analysis'
        full = '{0}/{1}'.format(base, verpath)

        if galaxy.release == 'MPL-4':
            assert 'default' in path
        else:
            assert 'MAPS' in path

        assert full in path
