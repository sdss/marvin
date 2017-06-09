#!/usr/bin/env python
# encoding: utf-8
#
# test_modelcube.py
#
# Created by José Sánchez-Gallego on 25 Sep 2016.


from __future__ import division, print_function, absolute_import

import os
import unittest

import pytest
from astropy.io import fits
from astropy.wcs import WCS

from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube


# class TestModelCubeBase(object):
#     """Defines the files and plateifus we will use in the tests."""
# 
#     @classmethod
#     def setUpClass(cls):
# 
#         super(TestModelCubeBase, cls).setUpClass()
#         #config.switchSasUrl('local')
#         cls.set_sasurl('local')
#         cls.release = 'MPL-5'
#         cls._update_release(cls.release)
#         cls.set_filepaths()
#         cls.filename = os.path.realpath(cls.modelpath)
# 
#     @classmethod
#     def tearDownClass(cls):
#         pass
# 
#     def setUp(self):
#         self._reset_the_config()
#         self._update_release(self.release)
#         self.set_filepaths()
#         assert os.path.exists(self.filename)
# 
#     def tearDown(self):
#         pass


class TestModelCubeInit(object):

    def _test_init(self, model_cube, galaxy, bintype='SPX', template_kin='GAU-MILESHC'):
        assert model_cube._release == galaxy.release
        assert model_cube._drpver == galaxy.drpver
        assert model_cube._dapver == galaxy.dapver
        assert model_cube.bintype == bintype
        assert model_cube.template_kin == template_kin
        assert model_cube.plateifu == galaxy.plateifu
        assert model_cube.mangaid == galaxy.mangaid
        assert isinstance(model_cube.header, fits.Header)
        assert isinstance(model_cube.wcs, WCS)
        assert model_cube.wavelength is not None
        assert model_cube.redcorr is not None

    @pytest.mark.parametrize('data_origin', ['file', 'db', 'api'])
    def test_init_modelcube(self, galaxy, data_origin):
        if data_origin == 'file':
            kwargs = {'filename': galaxy.modelpath}
        elif data_origin == 'db':
            kwargs = {'plateifu': galaxy.plateifu}
        elif data_origin == 'api':
            kwargs = {'plateifu': galaxy.plateifu, 'mode': 'remote'}

        model_cube = ModelCube(**kwargs)
        assert model_cube.data_origin == data_origin
        self._test_init(model_cube, galaxy)

    def test_init_from_file_global_mpl4(self, galaxy):
        mpl_ = config.release
        config.setMPL('MPL-4')
        model_cube = ModelCube(filename=galaxy.modelpath)
        assert model_cube.data_origin == 'file'
        self._test_init(model_cube, galaxy)
        config.setMPL(mpl_)

    def test_raises_exception_mpl4(self, galaxy):
        config.setMPL('MPL-4')
        with pytest.raises(MarvinError) as cm:
            ModelCube(plateifu=galaxy.plateifu)
        assert 'ModelCube requires at least dapver=\'2.0.2\'' in str(cm.value)

    @pytest.mark.parametrize('data_origin', ['file', 'db', 'api'])
    def test_init_modelcube_bintype(self, galaxy, data_origin):
        kwargs = {'bintype': galaxy.bintype}
        if data_origin == 'file':
            kwargs['filename'] = galaxy.modelpath
        elif data_origin == 'db':
            kwargs['plateifu'] = galaxy.plateifu
        elif data_origin == 'api':
            kwargs['plateifu'] = galaxy.plateifu
            kwargs['mode'] = 'remote'

        model_cube = ModelCube(**kwargs)
        assert model_cube.data_origin == data_origin
        self._test_init(model_cube, galaxy, bintype=galaxy.bintype)

    def test_get_flux_db(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu)
        assert model_cube.flux.shape == (4563, 34, 34)

    def test_get_flux_api_raises_exception(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        with pytest.raises(MarvinError) as cm:
            model_cube.flux
        assert 'cannot return a full cube in remote mode.' in str(cm.value)

    def test_get_cube_file(self, galaxy):
        model_cube = ModelCube(filename=galaxy.modelpath)
        assert isinstance(model_cube.cube, Cube)

    def test_get_maps_api(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        assert isinstance(model_cube.maps, Maps)


class TestGetSpaxel(object):

    def _test_getspaxel(self, spaxel, galaxy, bintype='SPX', template_kin='GAU-MILESHC'):
        assert spaxel._drpver == galaxy.drpver
        assert spaxel._dapver == galaxy.dapver
        assert spaxel.plateifu == galaxy.plateifu
        assert spaxel.mangaid == galaxy.mangaid
        assert spaxel.modelcube is not None
        assert spaxel.modelcube.bintype == bintype
        assert spaxel.modelcube.template_kin == template_kin
        assert spaxel._parent_shape == (34, 34)

        assert spaxel.model_flux is not None
        assert spaxel.model is not None
        assert spaxel.emline is not None
        assert spaxel.emline_base is not None
        assert spaxel.stellar_continuum is not None
        assert spaxel.redcorr is not None

    @pytest.mark.parametrize('data_origin', ['file', 'db','api'])
    def test_getspaxel(self, galaxy, data_origin):
        if data_origin == 'file':
            kwargs = {'filename': galaxy.modelpath}
        elif data_origin == 'db':
            kwargs = {'plateifu': galaxy.plateifu}
        elif data_origin == 'api':
            kwargs = {'plateifu': galaxy.plateifu, 'mode': 'remote'}

        model_cube = ModelCube(**kwargs)
        spaxel = model_cube.getSpaxel(x=1, y=2)
        self._test_getspaxel(spaxel, galaxy)

    def test_getspaxel_db_only_model(self, galaxy):

        model_cube = ModelCube(plateifu=galaxy.plateifu)
        spaxel = model_cube.getSpaxel(x=1, y=2, properties=False, spectrum=False)
        self._test_getspaxel(spaxel, galaxy)
        assert spaxel.cube is None
        assert spaxel.spectrum is None
        assert spaxel.maps is None
        assert len(spaxel.properties) == 0

    def test_getspaxel_matches_file_db_remote(self, galaxy):

        config.setMPL('MPL-5')
        assert config.release == 'MPL-5'

        # TODO abstact this out into fixtures........................................................
        modelcube_file = ModelCube(filename=galaxy.modelpath)
        modelcube_db = ModelCube(plateifu=galaxy.plateifu)
        modelcube_api = ModelCube(plateifu=galaxy.plateifu, mode='remote')

        assert modelcube_file.data_origin == 'file'
        assert modelcube_db.data_origin == 'db'
        assert modelcube_api.data_origin == 'api'

        xx = 12
        yy = 5
        spec_idx = 200

        spaxel_slice_file = modelcube_file[yy, xx]
        spaxel_slice_db = modelcube_db[yy, xx]
        spaxel_slice_api = modelcube_api[yy, xx]

        flux_result = 0.016027471050620079
        ivar_result = 361.13595581054693
        mask_result = 33

        assert pytest.approx(spaxel_slice_file.model_flux.flux[spec_idx], flux_result)
        assert pytest.approx(spaxel_slice_db.model_flux.flux[spec_idx], flux_result)
        assert pytest.approx(spaxel_slice_api.model_flux.flux[spec_idx], flux_result)

        assert pytest.approx(spaxel_slice_file.model_flux.ivar[spec_idx], ivar_result)
        assert pytest.approx(spaxel_slice_db.model_flux.ivar[spec_idx], ivar_result)
        assert pytest.approx(spaxel_slice_api.model_flux.ivar[spec_idx], ivar_result)

        assert pytest.approx(spaxel_slice_file.model_flux.mask[spec_idx], mask_result)
        assert pytest.approx(spaxel_slice_db.model_flux.mask[spec_idx], mask_result)
        assert pytest.approx(spaxel_slice_api.model_flux.mask[spec_idx], mask_result)

        xx_cen = -5
        yy_cen = -12

        spaxel_getspaxel_file = modelcube_file.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_db = modelcube_db.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_api = modelcube_api.getSpaxel(x=xx_cen, y=yy_cen)

        assert pytest.approx(spaxel_getspaxel_file.model_flux.flux[spec_idx], flux_result)
        assert pytest.approx(spaxel_getspaxel_db.model_flux.flux[spec_idx], flux_result)
        assert pytest.approx(spaxel_getspaxel_api.model_flux.flux[spec_idx], flux_result)

        assert pytest.approx(spaxel_getspaxel_file.model_flux.ivar[spec_idx], ivar_result)
        assert pytest.approx(spaxel_getspaxel_db.model_flux.ivar[spec_idx], ivar_result)
        assert pytest.approx(spaxel_getspaxel_api.model_flux.ivar[spec_idx], ivar_result)

        assert pytest.approx(spaxel_getspaxel_file.model_flux.mask[spec_idx], mask_result)
        assert pytest.approx(spaxel_getspaxel_db.model_flux.mask[spec_idx], mask_result)
        assert pytest.approx(spaxel_getspaxel_api.model_flux.mask[spec_idx], mask_result)


class TestPickling(object):

    def setUp(self):
        super(TestPickling, self).setUp()
        self._files_created = []

    def tearDown(self):

        super(TestPickling, self).tearDown()

        for fp in self._files_created:
            if os.path.exists(fp):
                os.remove(fp)

    def test_pickling_file(self):

        modelcube = ModelCube(filename=self.filename)
        assert modelcube.data_origin == 'file'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is not None

        path = modelcube.save()
        self._files_created.append(path)

        assert os.path.exists(path)
        assert os.path.realpath(path) == \
                         os.path.realpath(self.filename[0:-7] + 'mpf')
        assert modelcube.data is not None

        modelcube = None
        assert modelcube is None

        modelcube_restored = ModelCube.restore(path)
        assert modelcube_restored.data_origin == 'file'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is not None

    def test_pickling_file_custom_path(self):

        modelcube = ModelCube(filename=self.filename)

        test_path = '~/test.mpf'
        path = modelcube.save(path=test_path)
        self._files_created.append(path)

        assert os.path.exists(path)
        assert path == os.path.realpath(os.path.expanduser(test_path))

        modelcube_restored = ModelCube.restore(path, delete=True)
        assert modelcube_restored.data_origin == 'file'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is not None

        assert not os.path.exists(path)

    def test_pickling_db(self):

        modelcube = ModelCube(plateifu=self.plateifu)

        with pytest.raises(MarvinError) as ee:
            modelcube.save()

        assert 'objects with data_origin=\'db\' cannot be saved.' in \
                      str(ee.exception)

    def test_pickling_api(self):

        modelcube = ModelCube(plateifu=self.plateifu, mode='remote')
        assert modelcube.data_origin == 'api'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is None

        path = modelcube.save()
        self._files_created.append(path)

        assert os.path.exists(path)
        assert os.path.realpath(path) == \
                         os.path.realpath(self.filename[0:-7] + 'mpf')

        modelcube = None
        assert modelcube is None

        modelcube_restored = ModelCube.restore(path)
        assert modelcube_restored.data_origin == 'api'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is None
        assert modelcube_restored.header['VERSDRP3'] == 'v2_0_1'


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
