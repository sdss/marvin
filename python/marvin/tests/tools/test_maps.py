#!/usr/bin/env python
# encoding: utf-8
#
# test_maps.py
#
# Created by José Sánchez-Gallego on 22 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import unittest

import astropy.io.fits
import numpy as np

import marvin
import marvin.tests
import marvin.tools.map
import marvin.tools.maps
import marvin.tools.spaxel


class TestMapsBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    @classmethod
    def setUpClass(cls):

        marvin.config.switchSasUrl('local')

        cls.drpver_out = 'v1_5_1'
        cls.dapver_out = '1.1.1'

        cls.plate = 8485
        cls.mangaid = '1-209232'
        cls.plateifu = '8485-1901'
        cls.ifu = cls.plateifu.split('-')[1]
        cls.filename_default = os.path.join(
            os.getenv('MANGA_SPECTRO_ANALYSIS'), cls.drpver_out, cls.dapver_out,
            'full', str(cls.plate), str(cls.ifu),
            'manga-{0}-LOGCUBE_MAPS-NONE-013.fits.gz'.format(cls.plateifu))

        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.bintype = 'NONE'

        cls.marvindb_session = marvin.marvindb.session

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        marvin.marvindb.session = self.marvindb_session
        marvin.config.setMPL('MPL-4')
        self.assertTrue(os.path.exists(self.filename_default))

    def tearDown(self):
        pass

    def _assert_maps(self, maps):
        """Basic checks for a Maps object."""

        self.assertIsNotNone(maps)
        self.assertEqual(maps.plateifu, self.plateifu)
        self.assertEqual(maps.mangaid, self.mangaid)
        self.assertIsNotNone(maps.wcs)
        self.assertEqual(maps.bintype, self.bintype)
        self.assertListEqual(list(maps.shape), [34, 34])


class TestMapsFile(TestMapsBase):

    def test_load_from_file(self):

        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        self._assert_maps(maps)
        self.assertIsNotNone(maps.data)
        self.assertIsInstance(maps.data, astropy.io.fits.HDUList)
        self.assertIsNotNone(maps.cube)
        self.assertEqual(maps.cube.plateifu, self.plateifu)
        self.assertEqual(maps.cube.mangaid, self.mangaid)

    def test_load_file_mpl4_global_mpl5(self):

        marvin.config.setMPL('MPL-5')
        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        self.assertEqual(maps._release, 'MPL-4')
        self.assertEqual(maps._drpver, 'v1_5_1')
        self.assertEqual(maps._dapver, '1.1.1')

    def test_get_spaxel_file(self):

        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

        self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)

    def test_get_spaxel_test2_file(self):

        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        spaxel = maps.getSpaxel(x=5, y=5)

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

    def test_get_spaxel_no_db(self):
        """Tests getting an spaxel if there is no DB."""

        marvin.marvindb.session = None

        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        spaxel = maps.getSpaxel(x=5, y=5)

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

    def test_get_spaxel_binned_maps(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, bintype='VOR10',
                                      release='MPL-5')
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

        self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 0.00031520479546875247)
        self.assertEqual(spaxel.bintype, 'SPX')


class TestMapsDB(TestMapsBase):

    def test_load_default_from_db(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')
        self._assert_maps(maps)
        self.assertIsNotNone(maps.data)
        self.assertIsInstance(maps.data, marvin.marvindb.dapdb.File)
        self.assertIsNotNone(maps.cube)
        self.assertEqual(maps.cube.plateifu, self.plateifu)
        self.assertEqual(maps.cube.mangaid, self.mangaid)

    def test_load_full_from_db(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu,
                                      bintype='none', template_kin='MILES-THIN', mode='local')
        self._assert_maps(maps)
        self.assertIsNotNone(maps.data)
        self.assertIsInstance(maps.data, marvin.marvindb.dapdb.File)
        self.assertTrue(maps.bintype, 'NONE')
        self.assertTrue(maps.template_kin, 'MILES-THIN')

    def test_get_spaxel_db(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local', release='MPL-4')
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower', spectrum=False)

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

        self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)

    def test_get_spaxel_test2_db(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')
        spaxel = maps.getSpaxel(x=5, y=5)

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

    def test_get_spaxel_getitem(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')
        spaxel_getitem = maps[15, 8]

        self.assertTrue(isinstance(spaxel_getitem, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel_getitem.spectrum)
        self.assertTrue(len(spaxel_getitem.properties.keys()) > 0)

        self.assertAlmostEqual(spaxel_getitem.spectrum.flux[100], spaxel.spectrum.flux[100])


class TestMapsAPI(TestMapsBase):

    def test_load_default_from_api(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        self._assert_maps(maps)
        self.assertIsNone(maps.data)
        self.assertIsNotNone(maps.cube)
        self.assertEqual(maps.cube.data_origin, 'db')
        self.assertEqual(maps.cube.plateifu, self.plateifu)
        self.assertEqual(maps.cube.mangaid, self.mangaid)

    def test_load_full_from_api(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu,
                                      bintype='none', template_kin='MILES-THIN', mode='remote')
        self._assert_maps(maps)
        self.assertIsNone(maps.data)
        self.assertEqual(maps.data_origin, 'api')
        self.assertTrue(maps.bintype, 'NONE')
        self.assertTrue(maps.template_kin, 'MILES-THIN')

    def test_get_spaxel_api(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote', release='MPL-4')
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

        self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)

    def test_get_spaxel_test2_api(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        spaxel = maps.getSpaxel(x=5, y=5)

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNotNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

    def test_get_spaxel_drp_differ_from_global_api(self):

        marvin.config.setMPL('MPL-5')

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote', release='MPL-4')
        spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower', spectrum=False)

        self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
        self.assertIsNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties.keys()) > 0)

        self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)


class TestGetMap(TestMapsBase):

    def test_getmap_from_db(self):
        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')
        self.assertEqual(maps.data_origin, 'db')

        map_db = maps.getMap('specindex', channel='fe5406')
        self.assertIsInstance(map_db, marvin.tools.map.Map)
        self.assertIsInstance(map_db.header, astropy.io.fits.Header)
        self.assertEqual(map_db.header['C01'], 'D4000')

    def test_getmap_from_file(self):
        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        self.assertEqual(maps.data_origin, 'file')

        map_file = maps.getMap('specindex', channel='fe5406')
        self.assertIsInstance(map_file, marvin.tools.map.Map)
        self.assertIsInstance(map_file.header, astropy.io.fits.Header)
        self.assertEqual(map_file.header['C01'], 'D4000')

    def test_getmap_compare_db_file(self):

        maps_file = marvin.tools.maps.Maps(filename=self.filename_default)
        maps_db = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')

        residuals = (maps_file.getMap('emline_gflux', channel='oi_6365').value -
                     maps_db.getMap('emline_gflux', channel='oi_6365').value)

        self.assertAlmostEqual(np.sum(residuals), 0.0, places=5)

    def test_getmap_from_api(self):
        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        self.assertEqual(maps.data_origin, 'api')

        map_api = maps.getMap('specindex', channel='fe5406')
        self.assertIsInstance(map_api, marvin.tools.map.Map)
        self.assertIsInstance(map_api.header, astropy.io.fits.Header)
        self.assertEqual(map_api.header['C01'], 'D4000')

    def test_getmap_getitem(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu)

        map_getitem = maps['specindex_fe5406']
        self.assertIsInstance(map_getitem, marvin.tools.map.Map)

        map_getitem_no_channel = maps['binid']
        self.assertIsInstance(map_getitem_no_channel, marvin.tools.map.Map)


class TestPickling(TestMapsBase):

    def setUp(self):
        super(TestPickling, self).setUp()
        self._files_created = []

    def tearDown(self):

        super(TestPickling, self).tearDown()

        for fp in self._files_created:
            if os.path.exists(fp):
                os.remove(fp)

    def test_pickling_file(self):

        maps = marvin.tools.maps.Maps(filename=self.filename_default)
        self.assertEqual(maps.data_origin, 'file')
        self.assertIsInstance(maps, marvin.tools.maps.Maps)
        self.assertIsNotNone(maps.data)

        path = maps.save()
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.realpath(path),
                         os.path.realpath(self.filename_default[0:-7] + 'mpf'))
        self.assertIsNotNone(maps.data)

        maps = None
        self.assertIsNone(maps)

        maps_restored = marvin.tools.maps.Maps.restore(path)
        self.assertEqual(maps_restored.data_origin, 'file')
        self.assertIsInstance(maps_restored, marvin.tools.maps.Maps)
        self.assertIsNotNone(maps_restored.data)

        mm = maps_restored.getMap('emline_gflux', channel='ha_6564')
        self.assertIsInstance(mm, marvin.tools.map.Map)

    def test_pickling_file_custom_path(self):

        maps = marvin.tools.maps.Maps(filename=self.filename_default)

        test_path = '~/test.mpf'
        path = maps.save(path=test_path)
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(path, os.path.realpath(os.path.expanduser(test_path)))

        maps_restored = marvin.tools.maps.Maps.restore(path, delete=True)
        self.assertEqual(maps_restored.data_origin, 'file')
        self.assertIsInstance(maps_restored, marvin.tools.maps.Maps)
        self.assertIsNotNone(maps_restored.data)

        self.assertFalse(os.path.exists(path))

    def test_pickling_db(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu)

        with self.assertRaises(marvin.core.exceptions.MarvinError) as ee:
            maps.save()

        self.assertIn('objects with data_origin=\'db\' cannot be saved.',
                      str(ee.exception))

    def test_pickling_api(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        self.assertEqual(maps.data_origin, 'api')
        self.assertIsInstance(maps, marvin.tools.maps.Maps)
        self.assertIsNone(maps.data)

        path = maps.save()
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.realpath(path),
                         os.path.realpath(self.filename_default[0:-7] + 'mpf'))

        maps = None
        self.assertIsNone(maps)

        maps_restored = marvin.tools.maps.Maps.restore(path)
        self.assertEqual(maps_restored.data_origin, 'api')
        self.assertIsInstance(maps_restored, marvin.tools.maps.Maps)
        self.assertIsNone(maps_restored.data)
        self.assertEqual(maps_restored.header['VERSDRP3'], 'v1_5_0')


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
