# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-01 17:41:51
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-08 19:09:42

from __future__ import print_function, division, absolute_import

import unittest
import os
import glob
import warnings
from marvin import config
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.tools.plate import Plate
from marvin.tests import MarvinTest, skipIfNoBrian
from marvin.utils.general.images import getImagesByList, getImagesByPlate, getRandomImages, getDir3d, showImage

try:
    from sdss_access import RsyncAccess, AccessError
except ImportError:
    Path = None
    RsyncAccess = None


class TestImagesBase(MarvinTest):

    @classmethod
    def setUpClass(cls):

        super(TestImagesBase, cls).setUpClass()
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.ifu = '1901'
        cls.plateifu = '8485-1901'
        cls.mastar_plateifu = '8705-1901'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.imagelist = ['8485-1901', '7443-12701', '7443-1901']

        cls.new_plateifu = '7495-1901'
        cls.new_plate = 7495
        cls.new_ifu = '1901'
        cls.new_file = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.new_plateifu)

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
        self._remove_cube(release='MPL-5')
        self._remove_cube(release='MPL-4')

    def _update_release(self, release):
        config.setMPL(release)
        self.drpver, __ = config.lookUpVersions(release=release)

    def _make_paths(self, basepath, mode=None, inputs=None):
        fullpaths = []
        inputs = self.imagelist if not inputs else inputs
        for plateifu in inputs:
            plateid, ifu = plateifu.split('-')
            dir3d = getDir3d(plateifu, mode=mode)
            thepath = os.path.join(basepath, self.drpver, plateid, dir3d, 'images', ifu + '.png')
            fullpaths.append(thepath)
        return fullpaths

    def _get_cube(self, release=None):
        if release:
            self._update_release(release)
        filepath = os.path.join(self.mangaredux, self.drpver, str(self.new_plate), 'stack', self.new_file)
        if not os.path.isfile(filepath):
            rsync_access = RsyncAccess(label='marvin_getlist', verbose=False)
            rsync_access.remote()
            rsync_access.add('mangacube', plate=self.new_plate, drpver=self.drpver, ifu=self.new_ifu, dir3d='stack')
            rsync_access.set_stream()
            rsync_access.commit()

    def _remove_cube(self, release=None):
        if release:
            self._update_release(release)
        filepath = os.path.join(self.mangaredux, self.drpver, str(self.new_plate), 'stack', self.new_file)
        if os.path.isfile(filepath):
            os.remove(filepath)


class TestGetDir3d(TestImagesBase):

    def _getdir3d(self, expval, mode=None, plateifu=None):
        plateifu = self.plateifu if not plateifu else plateifu
        dir3d = getDir3d(plateifu, mode=mode, release=config.release)
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
        self._update_release('MPL-4')
        config.forceDbOff()
        self._getdir3d('stack', mode='remote', plateifu=self.new_plate)

    @unittest.SkipTest
    def test_getdir3d_remote_newplate_fail(self):
        self._update_release('MPL-5')
        config.forceDbOff()
        errmsg = 'Could not retrieve a remote plate.  If it is a mastar'
        with self.assertRaises(MarvinError) as cm:
            self._getdir3d('stack', mode='remote', plateifu=self.new_plate)
        self.assertIn(errmsg, str(cm.exception))

    def test_getdir3d_auto_newplate(self):
        self._update_release('MPL-4')
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

    def test_notvalid_mode(self):
        errmsg = 'Mode must be either auto, local, or remote'
        with self.assertRaises(AssertionError) as cm:
            image = getImagesByPlate(self.plate, mode='notvalidmode')
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
        self._update_release('MPL-4')
        config.forceDbOff()
        self._get_cube()
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


class TestRandomImages(TestImagesBase):

    def test_notvalid_mode(self):
        errmsg = 'Mode must be either auto, local, or remote'
        with self.assertRaises(AssertionError) as cm:
            image = getRandomImages(mode='notvalidmode')
        self.assertIn(errmsg, str(cm.exception))

    def test_get_images_download_local_fail(self):
        localpath = self._make_paths(self.mangaredux, mode='local', inputs=[self.new_plateifu])
        remotepath = self._make_paths(self.remoteredux, mode='remote', inputs=[self.new_plateifu])
        self.assertFalse(os.path.isfile(localpath[0]))
        errmsg = 'Download not available when in local mode'
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            image = getRandomImages(mode='local', as_url=True, download=True)
        self.assertIs(cm[-1].category, MarvinUserWarning)
        self.assertIn(errmsg, str(cm[-1].message))

    def _get_image_random(self, basedir, num=10, mode=None, as_url=None):
        images = getRandomImages(num=num, mode=mode, as_url=as_url)
        self.assertIn(basedir, images[0])
        self.assertIsInstance(images, list)
        self.assertIsNotNone(images)
        self.assertEqual(num, len(images))

    def test_get_images_auto(self):
        mode = 'auto'
        self._get_image_random(self.mangaredux, mode=mode)

    def test_get_images_local(self):
        mode = 'local'
        self._get_image_random(self.mangaredux, mode=mode)

    def test_get_images_local_num5(self):
        mode = 'local'
        self._get_image_random(self.mangaredux, num=5, mode=mode)

    def test_get_images_local_url(self):
        mode = 'local'
        self._get_image_random(self.remoteurl, mode=mode, as_url=True)

    def test_get_images_remote_url(self):
        mode = 'remote'
        self._get_image_random(self.remoteredux, mode=mode, as_url=True)

    def test_get_images_remote(self):
        mode = 'remote'
        self._get_image_random(self.mangaredux, mode=mode)

    def test_get_images_remote_num5(self):
        mode = 'remote'
        self._get_image_random(self.mangaredux, num=5, mode=mode)


class TestShowImage(TestImagesBase):
    def _show_image(self, path=None, plateifu=None, mode=None, release=None, return_image=True, show_image=None):
        image = showImage(path=path, plateifu=plateifu, mode=mode, release=release,
                          return_image=return_image, show_image=show_image)
        if return_image:
            self.assertIsNotNone(image)
        else:
            self.assertIsNone(image)
        return image

    def test_notvalid_mode(self):
        errmsg = 'Mode must be either auto, local, or remote'
        with self.assertRaises(AssertionError) as cm:
            self._show_image(mode='notvalidmode')
        self.assertIn(errmsg, str(cm.exception))

    def test_noinput(self):
        errmsg = 'A filepath or plateifu must be specified!'
        with self.assertRaises(AssertionError) as cm:
            self._show_image()
        self.assertIn(errmsg, str(cm.exception))

    # def test_mode_remote(self):
    #     errmsg = 'showImage currently only works in local mode.'
    #     with self.assertRaises(MarvinError) as cm:
    #         self._show_image(plateifu=self.plateifu, mode='remote')
    #     self.assertIn(errmsg, str(cm.exception))

    def test_mode_auto(self):
        image = self._show_image(plateifu=self.plateifu, mode='auto')

    def test_noreturn(self):
        image = self._show_image(plateifu=self.plateifu, return_image=False)
        self.assertIsNone(image)

    def _plateifu_fail(self, badplateifu, errmsg, mode=None):
        with self.assertRaises(MarvinError) as cm:
            self._show_image(plateifu=badplateifu, mode=mode)
        self.assertIn(errmsg, str(cm.exception))

    def test_plateifu_fail_local(self):
        badplateifu = '8485-1905'
        errmsg = 'Error: No files found locally to match plateifu {0}'.format(badplateifu)
        self._plateifu_fail(badplateifu, errmsg, mode='local')

    def test_plateifu_fail_remote(self):
        badplateifu = '8485-1905'
        badfilepath = 'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v1_5_1/8485/stack/images/1905.png'
        errmsg = 'Error: remote filepath {0}'.format(badfilepath)
        self._plateifu_fail(badplateifu, errmsg, mode='remote')

    def test_plateifu_fail_auto(self):
        badplateifu = '8485-1905'
        badfilepath = 'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v1_5_1/8485/stack/images/1905.png'
        errmsg = 'Error: remote filepath {0}'.format(badfilepath)
        self._plateifu_fail(badplateifu, errmsg, mode='auto')

    def _plateifu_success(self, mode=None):
        image = self._show_image(plateifu=self.plateifu, mode=mode)
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (562, 562))
        self.assertEqual(image.format, 'PNG')
        self.assertIn(str(self.plate), image.filename)
        self.assertIn(self.ifu, image.filename)
        if mode == 'remote':
            self.assertIn('https://data.sdss.org/sas/', image.filename)

    def test_plateifu_success_local(self):
        self._plateifu_success(mode='local')

    def test_plateifu_success_remote(self):
        self._plateifu_success(mode='remote')

    def test_plateifu_success_auto(self):
        self._plateifu_success(mode='auto')

    def test_path_fails_toomany(self):
        paths = self._make_paths(self.mangaredux, mode='local')
        errmsg = 'showImage currently only works on a single input at a time'
        with self.assertRaises(MarvinError) as cm:
            self._show_image(path=paths)
        self.assertIn(errmsg, str(cm.exception))

    def _path_fails_wrongmode(self, path, errmsg, mode=None):
        with self.assertRaises(MarvinError) as cm:
            self._show_image(path=path, mode=mode)
        self.assertIn(errmsg, str(cm.exception))

    def test_path_fails_localhttp(self):
        paths = self._make_paths(self.remoteurl, mode='remote')
        errmsg = 'Remote url path not allowed in local mode'
        self._path_fails_wrongmode(paths[0], errmsg, mode='local')

    def test_path_fails_remoteuserdir(self):
        paths = self._make_paths(self.mangaredux, mode='local')
        errmsg = 'Local path not allowed in remote mode'
        self._path_fails_wrongmode(paths[0], errmsg, mode='remote')

    def _path_success(self, paths, mode=None):
        image = self._show_image(path=paths[0], mode=mode)
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (562, 562))
        self.assertEqual(image.format, 'PNG')
        self.assertIn(str(self.plate), image.filename)
        self.assertIn(self.ifu, image.filename)
        if mode == 'remote':
            self.assertIn('https://data.sdss.org/sas/', image.filename)

    def test_path_success_local(self):
        paths = self._make_paths(self.mangaredux, mode='local')
        self._path_success(paths, mode='local')

    def test_path_success_remote(self):
        paths = self._make_paths(self.remoteurl, mode='remote')
        self._path_success(paths, mode='remote')

    def test_path_success_auto(self):
        paths = self._make_paths(self.mangaredux, mode='auto')
        self._path_success(paths, mode='auto')

    def test_path_badfile(self):
        badfile = os.path.expanduser('~/test_image.png')
        errmsg = 'Error: local filepath {0} does not exist. '.format(badfile)
        with self.assertRaises(MarvinError) as cm:
            self._show_image(path=badfile)
        self.assertIn(errmsg, str(cm.exception))

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)



