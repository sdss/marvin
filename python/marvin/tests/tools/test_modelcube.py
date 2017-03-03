#!/usr/bin/env python
# encoding: utf-8
#
# test_modelcube.py
#
# Created by José Sánchez-Gallego on 25 Sep 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import unittest

from astropy.io import fits
from astropy.wcs import WCS

import marvin
import marvin.tests

from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube


class TestModelCubeBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    @classmethod
    def setUpClass(cls):

        super(TestModelCubeBase, cls).setUpClass()
        #marvin.config.switchSasUrl('local')
        cls.set_sasurl('local')
        cls.release = 'MPL-5'
        cls._update_release(cls.release)
        cls.set_filepaths()
        cls.filename = os.path.realpath(cls.modelpath)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._reset_the_config()
        self._update_release(self.release)
        self.set_filepaths()
        self.assertTrue(os.path.exists(self.filename))

    def tearDown(self):
        pass


class TestModelCubeInit(TestModelCubeBase):

    def _test_init(self, model_cube, bintype='SPX', template_kin='GAU-MILESHC'):

        self.assertEqual(model_cube._release, self.release)
        self.assertEqual(model_cube._drpver, self.drpver)
        self.assertEqual(model_cube._dapver, self.dapver)
        self.assertEqual(model_cube.bintype, bintype)
        self.assertEqual(model_cube.template_kin, template_kin)
        self.assertEqual(model_cube.plateifu, self.plateifu)
        self.assertEqual(model_cube.mangaid, self.mangaid)
        self.assertIsInstance(model_cube.header, fits.Header)
        self.assertIsInstance(model_cube.wcs, WCS)
        self.assertIsNotNone(model_cube.wavelength)
        self.assertIsNotNone(model_cube.redcorr)

    def test_init_from_file(self):

        model_cube = ModelCube(filename=self.filename)
        self.assertEqual(model_cube.data_origin, 'file')
        self._test_init(model_cube)

    def test_init_from_file_global_mpl4(self):

        marvin.config.setMPL('MPL-4')
        model_cube = ModelCube(filename=self.filename)
        self.assertEqual(model_cube.data_origin, 'file')
        self._test_init(model_cube)

    def test_init_from_db(self):

        model_cube = ModelCube(plateifu=self.plateifu)
        self.assertEqual(model_cube.data_origin, 'db')
        self._test_init(model_cube)

    def test_init_from_api(self):

        model_cube = ModelCube(plateifu=self.plateifu, mode='remote')
        self.assertEqual(model_cube.data_origin, 'api')
        self._test_init(model_cube)

    def test_raises_exception_mpl4(self):

        marvin.config.setMPL('MPL-4')
        with self.assertRaises(MarvinError) as err:
            ModelCube(plateifu=self.plateifu)
        self.assertIn('ModelCube requires at least dapver=\'2.0.2\'', str(err.exception))

    def test_init_from_db_not_default(self):

        model_cube = ModelCube(plateifu=self.plateifu, bintype='NRE')
        self.assertEqual(model_cube.data_origin, 'db')
        self._test_init(model_cube, bintype='NRE')

    def test_init_from_api_not_default(self):

        model_cube = ModelCube(plateifu=self.plateifu, bintype='NRE', mode='remote')
        self.assertEqual(model_cube.data_origin, 'api')
        self._test_init(model_cube, bintype='NRE')

    def test_get_flux_db(self):

        model_cube = ModelCube(plateifu=self.plateifu)
        self.assertTupleEqual(model_cube.flux.shape, (4563, 34, 34))

    def test_get_flux_api_raises_exception(self):

        model_cube = ModelCube(plateifu=self.plateifu, mode='remote')
        with self.assertRaises(MarvinError) as err:
            model_cube.flux
        self.assertIn('cannot return a full cube in remote mode.', str(err.exception))

    def test_get_cube_file(self):

        model_cube = ModelCube(filename=self.filename)
        self.assertIsInstance(model_cube.cube, Cube)

    def test_get_maps_api(self):

        model_cube = ModelCube(plateifu=self.plateifu, mode='remote')
        self.assertIsInstance(model_cube.maps, Maps)


class TestGetSpaxel(TestModelCubeBase):

    def _test_getspaxel(self, spaxel, bintype='SPX', template_kin='GAU-MILESHC'):

        self.assertEqual(spaxel._drpver, self.drpver)
        self.assertEqual(spaxel._dapver, self.dapver)
        self.assertEqual(spaxel.plateifu, self.plateifu)
        self.assertEqual(spaxel.mangaid, self.mangaid)
        self.assertIsNotNone(spaxel.modelcube)
        self.assertEqual(spaxel.modelcube.bintype, bintype)
        self.assertEqual(spaxel.modelcube.template_kin, template_kin)
        self.assertTupleEqual(spaxel._parent_shape, (34, 34))

        self.assertIsNotNone(spaxel.model_flux)
        self.assertIsNotNone(spaxel.model)
        self.assertIsNotNone(spaxel.emline)
        self.assertIsNotNone(spaxel.emline_base)
        self.assertIsNotNone(spaxel.stellar_continuum)
        self.assertIsNotNone(spaxel.redcorr)

    def test_getspaxel_file(self):

        model_cube = ModelCube(filename=self.filename)
        spaxel = model_cube.getSpaxel(x=1, y=2)
        self._test_getspaxel(spaxel)

    def test_getspaxel_db(self):

        model_cube = ModelCube(plateifu=self.plateifu)
        spaxel = model_cube.getSpaxel(x=1, y=2)
        self._test_getspaxel(spaxel)

    def test_getspaxel_api(self):

        model_cube = ModelCube(plateifu=self.plateifu, mode='remote')
        spaxel = model_cube.getSpaxel(x=1, y=2)
        self._test_getspaxel(spaxel)

    def test_getspaxel_db_only_model(self):

        model_cube = ModelCube(plateifu=self.plateifu)
        spaxel = model_cube.getSpaxel(x=1, y=2, properties=False, spectrum=False)
        self._test_getspaxel(spaxel)
        self.assertIsNone(spaxel.cube)
        self.assertIsNone(spaxel.spectrum)
        self.assertIsNone(spaxel.maps)
        self.assertEqual(len(spaxel.properties), 0)

    def test_getspaxel_matches_file_db_remote(self):

        self._update_release('MPL-5')
        self.assertEqual(marvin.config.release, 'MPL-5')

        modelcube_file = ModelCube(filename=self.filename)
        modelcube_db = ModelCube(plateifu=self.plateifu)
        modelcube_api = ModelCube(plateifu=self.plateifu, mode='remote')

        self.assertEqual(modelcube_file.data_origin, 'file')
        self.assertEqual(modelcube_db.data_origin, 'db')
        self.assertEqual(modelcube_api.data_origin, 'api')

        xx = 12
        yy = 5
        spec_idx = 200

        spaxel_slice_file = modelcube_file[xx, yy]
        spaxel_slice_db = modelcube_db[xx, yy]
        spaxel_slice_api = modelcube_api[xx, yy]

        flux_result = 0.016027471050620079
        ivar_result = 361.13595581054693
        mask_result = 33

        self.assertAlmostEqual(spaxel_slice_file.model_flux.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_slice_db.model_flux.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_slice_api.model_flux.flux[spec_idx], flux_result)

        self.assertAlmostEqual(spaxel_slice_file.model_flux.ivar[spec_idx], ivar_result, places=5)
        self.assertAlmostEqual(spaxel_slice_db.model_flux.ivar[spec_idx], ivar_result, places=3)
        self.assertAlmostEqual(spaxel_slice_api.model_flux.ivar[spec_idx], ivar_result, places=3)

        self.assertAlmostEqual(spaxel_slice_file.model_flux.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_slice_db.model_flux.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_slice_api.model_flux.mask[spec_idx], mask_result)

        xx_cen = -5
        yy_cen = -12

        spaxel_getspaxel_file = modelcube_file.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_db = modelcube_db.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_api = modelcube_api.getSpaxel(x=xx_cen, y=yy_cen)

        self.assertAlmostEqual(spaxel_getspaxel_file.model_flux.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_getspaxel_db.model_flux.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_getspaxel_api.model_flux.flux[spec_idx], flux_result)

        self.assertAlmostEqual(spaxel_getspaxel_file.model_flux.ivar[spec_idx],
                               ivar_result, places=5)
        self.assertAlmostEqual(spaxel_getspaxel_db.model_flux.ivar[spec_idx],
                               ivar_result, places=3)
        self.assertAlmostEqual(spaxel_getspaxel_api.model_flux.ivar[spec_idx],
                               ivar_result, places=3)

        self.assertAlmostEqual(spaxel_getspaxel_file.model_flux.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_getspaxel_db.model_flux.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_getspaxel_api.model_flux.mask[spec_idx], mask_result)


class TestPickling(TestModelCubeBase):

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
        self.assertEqual(modelcube.data_origin, 'file')
        self.assertIsInstance(modelcube, ModelCube)
        self.assertIsNotNone(modelcube.data)

        path = modelcube.save()
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.realpath(path),
                         os.path.realpath(self.filename[0:-7] + 'mpf'))
        self.assertIsNotNone(modelcube.data)

        modelcube = None
        self.assertIsNone(modelcube)

        modelcube_restored = ModelCube.restore(path)
        self.assertEqual(modelcube_restored.data_origin, 'file')
        self.assertIsInstance(modelcube_restored, ModelCube)
        self.assertIsNotNone(modelcube_restored.data)

    def test_pickling_file_custom_path(self):

        modelcube = ModelCube(filename=self.filename)

        test_path = '~/test.mpf'
        path = modelcube.save(path=test_path)
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(path, os.path.realpath(os.path.expanduser(test_path)))

        modelcube_restored = ModelCube.restore(path, delete=True)
        self.assertEqual(modelcube_restored.data_origin, 'file')
        self.assertIsInstance(modelcube_restored, ModelCube)
        self.assertIsNotNone(modelcube_restored.data)

        self.assertFalse(os.path.exists(path))

    def test_pickling_db(self):

        modelcube = ModelCube(plateifu=self.plateifu)

        with self.assertRaises(MarvinError) as ee:
            modelcube.save()

        self.assertIn('objects with data_origin=\'db\' cannot be saved.',
                      str(ee.exception))

    def test_pickling_api(self):

        modelcube = ModelCube(plateifu=self.plateifu, mode='remote')
        self.assertEqual(modelcube.data_origin, 'api')
        self.assertIsInstance(modelcube, ModelCube)
        self.assertIsNone(modelcube.data)

        path = modelcube.save()
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.realpath(path),
                         os.path.realpath(self.filename[0:-7] + 'mpf'))

        modelcube = None
        self.assertIsNone(modelcube)

        modelcube_restored = ModelCube.restore(path)
        self.assertEqual(modelcube_restored.data_origin, 'api')
        self.assertIsInstance(modelcube_restored, ModelCube)
        self.assertIsNone(modelcube_restored.data)
        self.assertEqual(modelcube_restored.header['VERSDRP3'], 'v2_0_1')


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
