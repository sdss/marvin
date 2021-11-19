#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-08-15
# @Filename: test_modelcube.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-08 15:08:26


from __future__ import absolute_import, division, print_function

import os

import pytest
import six
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube

from .. import marvin_test_if


@pytest.fixture(autouse=True)
def skipmpl4(galaxy):
    if galaxy.release == 'MPL-4':
        pytest.skip('No modelcubes in MPL-4')


def is_string(string):
    return isinstance(string, six.string_types)


class TestModelCubeInit(object):

    def _test_init(self, model_cube, galaxy, bintype=None, template=None):
        assert model_cube.release == galaxy.release
        assert model_cube._drpver == galaxy.drpver
        assert model_cube._dapver == galaxy.dapver
        assert model_cube.bintype.name == bintype if bintype else galaxy.bintype.name
        assert model_cube.template == template if template else galaxy.template
        assert model_cube.plateifu == galaxy.plateifu
        assert model_cube.mangaid == galaxy.mangaid
        assert isinstance(model_cube.header, fits.Header)
        assert isinstance(model_cube.wcs, WCS)
        assert model_cube._wavelength is not None
        assert model_cube._redcorr is not None

    def test_init_modelcube(self, galaxy, data_origin):
        if data_origin == 'file':
            kwargs = {'filename': galaxy.modelpath}
        elif data_origin == 'db':
            kwargs = {'plateifu': galaxy.plateifu}
        elif data_origin == 'api':
            kwargs = {'plateifu': galaxy.plateifu, 'mode': 'remote'}

        model_cube = ModelCube(**kwargs)
        assert model_cube.data_origin == data_origin
        assert model_cube.nsa.z == pytest.approx(galaxy.redshift)
        self._test_init(model_cube, galaxy)

    def test_init_from_file_global_mpl4(self, galaxy):
        model_cube = ModelCube(filename=galaxy.modelpath, release='MPL-4')
        assert model_cube.data_origin == 'file'
        self._test_init(model_cube, galaxy)

    def test_raises_exception_mpl4(self, galaxy):
        with pytest.raises(MarvinError) as cm:
            ModelCube(plateifu=galaxy.plateifu, release='MPL-4')
        assert 'ModelCube requires at least dapver=\'2.0.2\'' in str(cm.value)

    def test_init_modelcube_bintype(self, galaxy, data_origin):
        kwargs = {'bintype': galaxy.bintype.name}
        if data_origin == 'file':
            kwargs['filename'] = galaxy.modelpath
        elif data_origin == 'db':
            kwargs['plateifu'] = galaxy.plateifu
        elif data_origin == 'api':
            kwargs['plateifu'] = galaxy.plateifu
            kwargs['mode'] = 'remote'

        model_cube = ModelCube(**kwargs)
        assert model_cube.data_origin == data_origin
        self._test_init(model_cube, galaxy, bintype=galaxy.bintype.name)


class TestModelCube(object):

    @marvin_test_if(mark='include', galaxy={'plateifu': '8485-1901'})
    def test_get_flux_db(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu)
        shape = tuple([4563] + galaxy.shape)
        assert model_cube.binned_flux.shape == shape

    @marvin_test_if(mark='include', galaxy={'plateifu': '8485-1901'})
    def test_get_flux_remote(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        shape = tuple([4563] + galaxy.shape)
        assert model_cube.binned_flux.shape == shape

    def test_get_cube_file(self, galaxy):
        model_cube = ModelCube(filename=galaxy.modelpath)
        assert isinstance(model_cube.getCube(), Cube)

    def test_get_cube_units(self, galaxy):
        model_cube = ModelCube(filename=galaxy.modelpath)
        unit = '1E-17 erg/s/cm^2/ang/spaxel'
        fileunit = model_cube.data['EMLINE'].header['BUNIT']
        assert unit == fileunit

        unit = fileunit.replace('ang', 'angstrom').split('/spaxel')[0]
        spaxel = u.Unit('spaxel', represents=u.pixel,
                        doc='A spectral pixel', parse_strict='silent')
        newunit = (u.Unit(unit) / spaxel)
        #unit = '1e-17 erg / (Angstrom cm2 s spaxel)'
        dmunit = model_cube.emline_fit.unit
        assert newunit == dmunit

    def test_get_maps_api(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        assert isinstance(model_cube.getMaps(), Maps)

    def test_nobintype_in_db(self, galaxy):

        if galaxy.release != 'MPL-6':
            pytest.skip('only running this test for MPL6')

        with pytest.raises(MarvinError) as cm:
            ModelCube(plateifu=galaxy.plateifu, bintype='ALL', release=galaxy.release)

        assert 'Specified bintype ALL is not available in the DB' in str(cm.value)

    @pytest.mark.parametrize('objtype, errmsg',
                             [('cube', 'Trying to open a non DAP file with Marvin ModelCube'),
                              ('maps', 'Trying to open a DAP MAPS with Marvin ModelCube')])
    def test_modelcube_wrong_file(self, galaxy, objtype, errmsg):
        path = galaxy.cubepath if objtype == 'cube' else galaxy.mapspath
        with pytest.raises(MarvinError) as cm:
            ModelCube(filename=path)
        assert errmsg in str(cm.value)


class TestPickling(object):

    def test_pickling_file(self, temp_scratch, galaxy):
        modelcube = ModelCube(filename=galaxy.modelpath, bintype=galaxy.bintype)
        assert modelcube.data_origin == 'file'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is not None

        file = temp_scratch.join('test_modelcube.mpf')
        modelcube.save(str(file))

        assert file.check() is True
        assert modelcube.data is not None

        modelcube = None
        assert modelcube is None

        modelcube_restored = ModelCube.restore(str(file))
        assert modelcube_restored.data_origin == 'file'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is not None

    def test_pickling_file_custom_path(self, temp_scratch, galaxy):
        modelcube = ModelCube(filename=galaxy.modelpath, bintype=galaxy.bintype)
        assert modelcube.data_origin == 'file'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is not None

        file = temp_scratch.join('mcpickle').join('test_modelcube.mpf')
        assert file.check(file=1) is False

        path = modelcube.save(path=str(file))
        assert file.check() is True
        assert os.path.exists(path)

        modelcube_restored = ModelCube.restore(str(file), delete=True)
        assert modelcube_restored.data_origin == 'file'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is not None

        assert not os.path.exists(path)

    def test_pickling_db(self, galaxy, temp_scratch):
        modelcube = ModelCube(plateifu=galaxy.plateifu, bintype=galaxy.bintype)

        file = temp_scratch.join('test_modelcube_db.mpf')
        with pytest.raises(MarvinError) as cm:
            modelcube.save(str(file))

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_api(self, temp_scratch, galaxy):
        modelcube = ModelCube(plateifu=galaxy.plateifu, bintype=galaxy.bintype, mode='remote')
        assert modelcube.data_origin == 'api'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is None

        file = temp_scratch.join('test_modelcube_api.mpf')
        modelcube.save(str(file))

        assert file.check() is True

        modelcube = None
        assert modelcube is None

        modelcube_restored = ModelCube.restore(str(file))
        assert modelcube_restored.data_origin == 'api'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is None


class TestMaskbit(object):

    def test_quality_flag(self, galaxy):
        modelcube = ModelCube(plateifu=galaxy.plateifu, bintype=galaxy.bintype)
        assert modelcube.quality_flag is not None

    @pytest.mark.parametrize('flag',
                             ['manga_target1',
                              'manga_target2',
                              'manga_target3',
                              'target_flags'])
    def test_flag(self, flag, galaxy):
        modelcube = ModelCube(plateifu=galaxy.plateifu, bintype=galaxy.bintype)
        assert getattr(modelcube, flag, None) is not None
