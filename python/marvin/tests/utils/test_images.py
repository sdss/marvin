# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-01 17:41:51
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-01 23:11:25

from __future__ import print_function, division, absolute_import

import unittest
import os
import glob
import warnings
from marvin import config
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.tools.plate import Plate
from marvin.tests import MarvinTest, skipIfNoBrian
from marvin.utils.general.images import getImagesByList, getImagesByPlate, getRandomImages, getDir3d


class TestImagesBase(MarvinTest):

    @classmethod
    def setUpClass(cls):

        super(TestImagesBase, cls).setUpClass()
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.plateifu = '8485-1901'
        cls.mastar_plateifu = '8705-1901'
        cls.new_plateifu = '7495-1901'
        cls.new_plate = 7495
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.imagelist = ['8485-1901', '7443-12701', '7443-1901']

        cls.sasbasedir = os.getenv("$SAS_BASE_DIR")
        cls.mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
        cls.remoteredux = 'https://sdss@dtn01.sdss.org/sas/mangawork/manga/spectro/redux/'
        cls.remoteurl = 'https://data.sdss.org/sas/mangawork/manga/spectro/redux/'

        cls.init_mode = config.mode
        cls.init_sasurl = config.sasurl
        cls.init_urlmap = config.urlmap

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        config.switchSasUrl('local')
        config.sasurl = self.init_sasurl
        self.mode = self.init_mode
        config.urlmap = self.init_urlmap
        config.setMPL('MPL-4')
        config.forceDbOn()
        self.drpver, __ = config.lookUpVersions(release=config.release)

    def tearDown(self):
        plate, ifu = self.new_plateifu.split('-')
        newdir = os.path.join(self.mangaredux, self.drpver, plate, 'stack/images')
        newpath = os.path.join(newdir, '*.png')
        newfiles = glob.glob(newpath)
        for file in newfiles:
            if os.path.isfile(file):
                os.remove(file)

    def _update_release(self, release):
        config.setMPL(release)
        self.drpver, __ = config.lookUpVersions(release=release)

    def _make_paths(self, basepath, mode=None, inputs=None):
        fullpaths = []
        inputs = self.imagelist if not inputs else inputs
        for plateifu in inputs:
            plateid, ifu = plateifu.split('-')
            dir3d = getDir3d(plateifu, mode=mode)
            thepath = os.path.join(basepath, self.drpver, plateid, dir3d, 'images', ifu+'.png')
            fullpaths.append(thepath)
        return fullpaths


class TestGetDir3d(TestImagesBase):

    def _getdir3d(self, expval, mode=None, plateifu=None):
        plateifu = self.plateifu if not plateifu else plateifu
        dir3d = getDir3d(plateifu, mode=mode)
        self.assertEqual(expval, dir3d)

    def test_getdir3d_local(self):
        self._getdir3d('stack', mode='local')

    def test_getdir3d_remote(self):
        self._getdir3d('stack', mode='remote')

    def test_getdir3d_auto(self):
        self._getdir3d('stack', mode='auto')

    def test_getdir3d_mastar_local(self):
        self._update_release('MPL-5')
        self._getdir3d('stack', mode='local')

    def test_getdir3d_mpl5_remote(self):
        self._update_release('MPL-5')
        self._getdir3d('stack', mode='remote')

    def test_getdir3d_mpl5_auto(self):
        self._update_release('MPL-5')
        self._getdir3d('stack', mode='auto')

    def test_getdir3d_local_plate(self):
        self._update_release('MPL-5')
        self._getdir3d('stack', mode='local', plateifu=self.plate)

    def test_getdir3d_remote_plate(self):
        self._update_release('MPL-5')
        self._getdir3d('stack', mode='remote', plateifu=self.plate)

    def test_getdir3d_auto_plate(self):
        self._update_release('MPL-5')
        self._getdir3d('stack', mode='auto', plateifu=self.plate)

    def test_getdir3d_local_plate_nodb(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('stack', mode='local', plateifu=self.plate)

    def test_getdir3d_remote_plate_nodb(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('stack', mode='remote', plateifu=self.plate)

    def test_getdir3d_auto_plate_nodb(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('stack', mode='auto', plateifu=self.plate)

    def test_getdir3d_local_newplate_nocubes(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        errmsg = 'this is the end of the road. Try using some reasonable inputs.'
        with self.assertRaises(MarvinError) as cm:
            self._getdir3d('stack', mode='local', plateifu=self.new_plate)
        self.assertIn(errmsg, str(cm.exception))

    def test_getdir3d_remote_newplate(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('stack', mode='remote', plateifu=self.new_plate)

    def test_getdir3d_auto_newplate(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('stack', mode='auto', plateifu=self.new_plate)

    @unittest.SkipTest
    def test_getdir3d_mastar_local(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('mastar', mode='local', plateifu=self.mastar_plateifu)

    @unittest.SkipTest
    def test_getdir3d_mastar_remote(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('mastar', mode='remote', plateifu=self.mastar_plateifu)

    @unittest.SkipTest
    def test_getdir3d_mastar_auto(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        self._getdir3d('mastar', mode='auto', plateifu=self.mastar_plateifu)


class TestImagesByList(TestImagesBase):

    def test_notvalid_input(self):
        errmsg = 'Input must be of type list or Numpy array'
        with self.assertRaises(AssertionError) as cm:
            image = getImagesByList(self.new_plateifu, mode='local')
        self.assertIn(errmsg, str(cm.exception))

    def test_notvalid_objectid(self):
        errmsg = 'Input must be of type plate-ifu or mangaid'
        with self.assertRaises(AssertionError) as cm:
            image = getImagesByList(['nogoodid'], mode='local')
        self.assertIn(errmsg, str(cm.exception))

    def test_notvalid_mode(self):
        errmsg = 'Mode must be either auto, local, or remote'
        with self.assertRaises(AssertionError) as cm:
            image = getImagesByList(self.imagelist, mode='notvalidmode')
        self.assertIn(errmsg, str(cm.exception))

    def _get_imagelist(self, explist, inputlist=None, mode=None, as_url=None):
        images = getImagesByList(inputlist, mode=mode, as_url=as_url)
        self.assertListEqual(explist, images)

    def test_get_images_auto(self):
        mode = 'auto'
        paths = self._make_paths(self.mangaredux, mode=mode)
        self._get_imagelist(paths, inputlist=self.imagelist, mode=mode)

    def test_get_images_local(self):
        mode = 'local'
        paths = self._make_paths(self.mangaredux, mode=mode)
        self._get_imagelist(paths, inputlist=self.imagelist, mode=mode)

    def test_get_images_local_url(self):
        mode = 'local'
        paths = self._make_paths(self.remoteurl, mode=mode)
        self._get_imagelist(paths, inputlist=self.imagelist, mode=mode, as_url=True)

    def test_get_images_remote_url(self):
        mode = 'remote'
        paths = self._make_paths(self.remoteredux, mode=mode)
        self._get_imagelist(paths, inputlist=self.imagelist, mode=mode, as_url=True)

    def test_get_images_remote(self):
        mode = 'remote'
        paths = self._make_paths(self.mangaredux, mode=mode)
        self._get_imagelist(paths, inputlist=self.imagelist, mode=mode)

    def test_get_images_download_remote(self):
        localpath = self._make_paths(self.mangaredux, mode='local', inputs=[self.new_plateifu])
        remotepath = self._make_paths(self.remoteredux, mode='remote', inputs=[self.new_plateifu])
        self.assertFalse(os.path.isfile(localpath[0]))
        image = getImagesByList([self.new_plateifu], mode='remote', as_url=True, download=True)
        self.assertTrue(os.path.isfile(localpath[0]))
        self.assertIsNone(image)

    def test_get_images_download_local_fail(self):
        localpath = self._make_paths(self.mangaredux, mode='local', inputs=[self.new_plateifu])
        remotepath = self._make_paths(self.remoteredux, mode='remote', inputs=[self.new_plateifu])
        self.assertFalse(os.path.isfile(localpath[0]))
        errmsg = 'Download not available when in local mode'
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            image = getImagesByList([self.new_plateifu], mode='local', as_url=True, download=True)
        self.assertIs(cm[-1].category, MarvinUserWarning)
        self.assertIn(errmsg, str(cm[-1].message))


class TestImagesByPlate(TestImagesBase):

    def test_notvalid_plate(self):
        errmsg = 'Plateid must be a numeric integer value'
        with self.assertRaises(AssertionError) as cm:
            image = getImagesByPlate('8485abc', mode='local')
        self.assertIn(errmsg, str(cm.exception))

    def _get_imageplate(self, explist, plate=None, mode=None, as_url=None):
        images = getImagesByPlate(plate, mode=mode, as_url=as_url)
        self.assertIn(explist[0], images)

    def test_get_images_auto(self):
        mode = 'auto'
        paths = self._make_paths(self.mangaredux, mode=mode, inputs=[self.plateifu])
        self._get_imageplate(paths, plate=self.plate, mode=mode)

    def test_get_images_local(self):
        mode = 'local'
        paths = self._make_paths(self.mangaredux, mode=mode, inputs=[self.plateifu])
        self._get_imageplate(paths, plate=self.plate, mode=mode)

    def test_get_images_local_url(self):
        mode = 'local'
        paths = self._make_paths(self.remoteurl, mode=mode, inputs=[self.plateifu])
        self._get_imageplate(paths, plate=self.plate, mode=mode, as_url=True)

    def test_get_images_remote_url(self):
        mode = 'remote'
        paths = self._make_paths(self.remoteredux, mode=mode, inputs=[self.plateifu])
        self._get_imageplate(paths, plate=self.plate, mode=mode, as_url=True)

    def test_get_images_remote(self):
        mode = 'remote'
        paths = self._make_paths(self.mangaredux, mode=mode, inputs=[self.plateifu])
        self._get_imageplate(paths, plate=self.plate, mode=mode)

    def test_get_images_download_remote(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        localpath = self._make_paths(self.mangaredux, mode='local', inputs=[self.new_plateifu])
        remotepath = self._make_paths(self.remoteredux, mode='remote', inputs=[self.new_plateifu])
        self.assertFalse(os.path.isfile(localpath[0]))
        image = getImagesByPlate(self.new_plate, mode='remote', as_url=True, download=True)
        self.assertTrue(os.path.isfile(localpath[0]))
        self.assertIsNone(image)

    def test_get_images_download_local_fail(self):
        localpath = self._make_paths(self.mangaredux, mode='local', inputs=[self.new_plateifu])
        remotepath = self._make_paths(self.remoteredux, mode='remote', inputs=[self.new_plateifu])
        self.assertFalse(os.path.isfile(localpath[0]))
        errmsg = 'Download not available when in local mode'
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            image = getImagesByPlate(self.new_plate, mode='local', as_url=True, download=True)
        self.assertIs(cm[-1].category, MarvinUserWarning)
        self.assertIn(errmsg, str(cm[-1].message))

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)



