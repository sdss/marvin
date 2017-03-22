#!/usr/bin/env python
# encoding: utf-8
#
# test_bin.py
#
# Created by José Sánchez-Gallego on 6 Nov 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest

import marvin
from marvin import config
import marvin.tests
import marvin.tools.bin
import marvin.tools.maps
import marvin.tools.modelcube
import marvin.utils.general
from marvin.core.exceptions import MarvinError


def setup_config(cls):
    config.use_sentry = False
    config.add_github_message = False

@pytest.fixture(scope='module')
def setup_config_vars():
    # set initial config variables
    init_mode = config.mode
    init_sasurl = config.sasurl
    init_urlmap = config.urlmap
    init_xyorig = config.xyorig
    init_traceback = config._traceback
    init_keys = ['mode', 'sasurl', 'urlmap', 'xyorig', 'traceback']
    return init_mode

@pytest.fixture(scope='module')
def setup_db():
    # set db stuff
    _marvindb = marvindb
    session = marvindb.session
    return session

@pytest.fixture(scope='module')
def setup_paths():
    # set paths
    sasbasedir = os.getenv("$SAS_BASE_DIR")
    mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
    mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")
    print('setup_paths')

@pytest.fixture(scope='module',
                params=dict(
                    plateifu='8485-1901',
                    mangaid='1-209232',
                    ra=232.544703894,
                    dec = 48.6902009334,
                    redshift=0.0407447,
                    dir3d='stack',
                    release='MPL-5',
                    cubename='manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu),
                    rssname='manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu),
                    imgname='{0}.png'.format(cls.ifu),
                    mapsname='manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp),
                    modelname='manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
                    ))

def params_8485_1901():

    # testing data for 8485-1901
    cls.set_plateifu(plateifu='8485-1901')
    cls.mangaid = '1-209232'
    cls.cubepk = 10179
    cls.ra = 232.544703894
    cls.dec = 48.6902009334
    cls.redshift = 0.0407447
    cls.dir3d = 'stack'
    cls.release = 'MPL-5'
    cls.drpver, cls.dapver = config.lookUpVersions(cls.release)  # NEED TO INCLUDE
    cls.bintemp = _get_bintemps(cls.dapver, default=True)  # NEED TO INCLUDE
    cls.defaultbin, cls.defaulttemp = cls.bintemp.split('-', 1)  # NEED TO INCLUDE
    cls.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu)
    cls.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu)
    cls.imgname = '{0}.png'.format(cls.ifu)
    cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
    cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)

def setUpClass(cls):
    super(TestBinBase, cls).setUpClass()
    cls._update_release('MPL-5')
    cls.set_sasurl('local')
    cls.set_filepaths(bintype='VOR10')
    cls.maps_filename = cls.mapspath
    cls.modelcube_filename = cls.modelpath


class MarvinTest(TestCase):
    """Custom class for Marvin-tools tests."""

    def skipTest(self, test):
        """Issues a warning when we skip a test."""
        warnings.warn('Skipped test {0} because there is no DB connection.'
                      .format(test.__name__), MarvinSkippedTestWarning)

    def skipBrian(self, test):
        """Issues a warning when we skip a test."""
        warnings.warn('Skipped test {0} because there is no Brian.'
                      .format(test.__name__), MarvinSkippedTestWarning)

    @classmethod
    def setUpClass(cls):
        config.use_sentry = False
        config.add_github_message = False

        # set initial config variables
        cls.init_mode = config.mode
        cls.init_sasurl = config.sasurl
        cls.init_urlmap = config.urlmap
        cls.init_xyorig = config.xyorig
        cls.init_traceback = config._traceback
        cls.init_keys = ['mode', 'sasurl', 'urlmap', 'xyorig', 'traceback']

        # set db stuff
        cls._marvindb = marvindb
        cls.session = marvindb.session

        # set paths
        cls.sasbasedir = os.getenv("$SAS_BASE_DIR")
        cls.mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
        cls.mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")

        # testing data for 8485-1901
        cls.set_plateifu(plateifu='8485-1901')
        cls.mangaid = '1-209232'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.redshift = 0.0407447
        cls.dir3d = 'stack'
        cls.release = 'MPL-5'
        cls.drpver, cls.dapver = config.lookUpVersions(cls.release)
        cls.bintemp = _get_bintemps(cls.dapver, default=True)
        cls.defaultbin, cls.defaulttemp = cls.bintemp.split('-', 1)
        cls.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu)
        cls.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu)
        cls.imgname = '{0}.png'.format(cls.ifu)
        cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
        cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)

    def _reset_the_config(self):
        keys = self.init_keys
        for key in keys:
            ikey = 'init_{0}'.format(key)
            if hasattr(self, ikey):
                k = '_{0}'.format(key) if 'traceback' in key else key
                config.__setattr__(k, self.__getattribute__(ikey))

    @classmethod
    def set_sasurl(cls, loc='local', port=5000):
        istest = True if loc == 'utah' else False
        config.switchSasUrl(loc, test=istest, port=port)
        response = Interaction('api/general/getroutemap', request_type='get')
        config.urlmap = response.getRouteMap()

    @classmethod
    def _update_release(cls, release):
        config.setMPL(release)
        cls.drpver, cls.dapver = config.lookUpVersions(release=release)

    @classmethod
    def update_names(cls, bintype=None, template=None):
        if not bintype:
            bintype = cls.defaultbin
        if not template:
            template = cls.defaulttemp

        cls.bintype = bintype
        cls.template = template
        cls.bintemp = '{0}-{1}'.format(bintype, template)
        cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
        cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)

    @classmethod
    def set_filepaths(cls, bintype=None, template=None):
        # Paths
        cls.drppath = os.path.join(cls.mangaredux, cls.drpver)
        cls.dappath = os.path.join(cls.mangaanalysis, cls.drpver, cls.dapver)
        cls.imgpath = os.path.join(cls.mangaredux, cls.drpver, str(cls.plate), cls.dir3d, 'images')

        # DRP filename paths
        cls.cubepath = os.path.join(cls.drppath, str(cls.plate), cls.dir3d, cls.cubename)
        cls.rsspath = os.path.join(cls.drppath, str(cls.plate), cls.dir3d, cls.rssname)

        # DAP filename paths
        if (bintype or template):
            cls.update_names(bintype=bintype, template=template)
        cls.analpath = os.path.join(cls.dappath, cls.bintemp, str(cls.plate), cls.ifu)
        cls.mapspath = os.path.join(cls.analpath, cls.mapsname)
        cls.modelpath = os.path.join(cls.analpath, cls.modelname)

    @classmethod
    def set_plateifu(cls, plateifu='8485-1901'):
        cls.plateifu = plateifu
        cls.plate, cls.ifu = cls.plateifu.split('-')
        cls.plate = int(cls.plate)


class TestBinBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    # @classmethod
    # def setUpClass(cls):
    #     super(TestBinBase, cls).setUpClass()
    #     cls._update_release('MPL-5')
    #     cls.set_sasurl('local')
    #     cls.set_filepaths(bintype='VOR10')
    #     cls.maps_filename = cls.mapspath
    #     cls.modelcube_filename = cls.modelpath

    # @classmethod
    # def tearDownClass(cls):
    #     pass

    def setUp(self):
        self._reset_the_config()
        self._update_release('MPL-5')
        self.set_sasurl('local')
        self.assertTrue(os.path.exists(self.maps_filename))
        self.assertTrue(os.path.exists(self.modelcube_filename))

    def tearDown(self):
        pass


class TestBinInit(TestBinBase):

    def _check_bin_data(self, bb):

        self.assertEqual(bb.binid, 100)
        self.assertEqual(bb.plateifu, self.plateifu)
        self.assertEqual(bb.mangaid, self.mangaid)

        self.assertTrue(len(bb.spaxels) == 2)
        self.assertFalse(bb.spaxels[0].loaded)

        self.assertIsNotNone(bb.properties)

    def test_init_from_files(self):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename,
                                  modelcube_filename=self.modelcube_filename)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb)

    def test_init_from_file_only_maps(self):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsNotNone(bb._modelcube)
        self.assertEqual(bb._modelcube.data_origin, 'db')
        self.assertEqual(bb._modelcube.bintype, self.bintype)

        self._check_bin_data(bb)

    def test_init_from_db(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, bintype=self.bintype)
        self.assertEqual(bb._maps.data_origin, 'db')
        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        self.assertEqual(bb._modelcube.bintype, self.bintype)

        self._check_bin_data(bb)

    def test_init_from_api(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, mode='remote',
                                  bintype=self.bintype)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        self.assertEqual(bb._modelcube.bintype, self.bintype)

        self._check_bin_data(bb)

    def test_bin_does_not_exist(self):

        with self.assertRaises(MarvinError) as ee:
            marvin.tools.bin.Bin(binid=99999, plateifu=self.plateifu, mode='local',
                                 bintype=self.bintype)
            self.assertIn('there are no spaxels associated with binid=99999.', str(ee.exception))


class TestBinFileMismatch(TestBinBase):

    @unittest.expectedFailure
    def test_bintypes(self):

        wrong_bintype = 'SPX'
        self.assertNotEqual(wrong_bintype, self.bintype)

        wrong_modelcube_filename = os.path.join(
            self.path_release,
            '{0}-GAU-MILESHC'.format(wrong_bintype), str(self.plate), str(self.ifu),
            'manga-{0}-{1}-{2}-GAU-MILESHC.fits.gz'.format(self.plateifu, 'LOGCUBE', wrong_bintype))

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename,
                                  modelcube_filename=wrong_modelcube_filename)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self.assertRaises(MarvinError,
                          marvin.utils.general._check_file_parameters(bb._maps, bb._modelcube))


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
