#!/usr/bin/env python
# encoding: utf-8
#
# conftest.py
#
# Created by Brett Andrews on 20 Mar 2017.


import os

import pytest

from marvin import config, marvindb
from marvin.api.api import Interaction
from marvin.tools.maps import _get_bintemps


# --------- #
#           #
# Version 2 #
#           #
# --------- #


class MarvinSession:
    """Custom class for Marvin-tools tests."""

    # necessary?
    def __init__(self, request):
        self.request = request

    # TODO: Replace instance of skipTest and skipBrian with pytest's skipif
    #
    # def skipTest(self, test):
    #     """Issues a warning when we skip a test."""
    #     warnings.warn('Skipped test {0} because there is no DB connection.'
    #                   .format(test.__name__), MarvinSkippedTestWarning)
    #
    # def skipBrian(self, test):
    #     """Issues a warning when we skip a test."""
    #     warnings.warn('Skipped test {0} because there is no Brian.'
    #                   .format(test.__name__), MarvinSkippedTestWarning)

    # TODO how to pass in classmethod without subclassing?
    # Can I do this as a session level fixture? Probably
    @classmethod
    def set_config(cls):
        config.use_sentry = False
        config.add_github_message = False

        # TODO use monkeypatch to safely set these attributes
        #
        # set initial config variables
        # cls.init_mode = config.mode
        # cls.init_sasurl = config.sasurl
        # cls.init_urlmap = config.urlmap
        # cls.init_xyorig = config.xyorig
        # cls.init_traceback = config._traceback
        # cls.init_keys = ['mode', 'sasurl', 'urlmap', 'xyorig', 'traceback']

    @classmethod
    def set_db(cls):
        cls._marvindb = marvindb
        cls.session = marvindb.session

    @classmethod
    def set_paths(cls):
        cls.sasbasedir = os.getenv("$SAS_BASE_DIR")
        cls.mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
        cls.mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")

        # TODO refactor into separate method (setup_8485_1901)
        #
        # testing data for 8485-1901
        # cls.set_plateifu(plateifu='8485-1901')
        # cls.mangaid = '1-209232'
        # cls.cubepk = 10179
        # cls.ra = 232.544703894
        # cls.dec = 48.6902009334
        # cls.redshift = 0.0407447
        # cls.dir3d = 'stack'
        # TODO add release setup session fixture
        cls.release = 'MPL-5'
        cls.drpver, cls.dapver = config.lookUpVersions(cls.release)
        # TODO add filename setup session fixture
        # cls.bintemp = _get_bintemps(cls.dapver, default=True)
        # cls.defaultbin, cls.defaulttemp = cls.bintemp.split('-', 1)
        # cls.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu)
        # cls.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu)
        # cls.imgname = '{0}.png'.format(cls.ifu)
        # cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
        # cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)

    # TODO replace with monkeypatch
    #
    # def _reset_the_config(self):
    #     keys = self.init_keys
    #     for key in keys:
    #         ikey = 'init_{0}'.format(key)
    #         if hasattr(self, ikey):
    #             k = '_{0}'.format(key) if 'traceback' in key else key
    #             config.__setattr__(k, self.__getattribute__(ikey))

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


class Galaxy(MarvinSession):
    """An example galaxy for Marvin-tools testing."""

    dir3d = 'stack'

    @classmethod
    def set_plateifu(cls, plateifu='8485-1901'):
        cls.plateifu = plateifu
        cls.plate, cls.ifu = cls.plateifu.split('-')
        cls.plate = int(cls.plate)


@pytest.fixture(scope='session')
def set_galaxy(plateifu='8485-1901'):
    plate, ifu = set_plateifu(plateifu=plateifu)
    galaxy = galaxies['plateifu']
    mangaid = galaxy['mangaid']
    cubepk = galaxy['cubepk']
    ra = galaxy['ra']
    dec = galaxy['dec']
    redshift = galaxy['redshift']
    return
    
    @classmethod
    def set_galaxy(cls, plateifu):
        # TODO refactor into separate method (setup_8485_1901)
        #
        # testing data for 8485-1901
        cls.set_plateifu(plateifu='8485-1901')
        cls.mangaid = '1-209232'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.redshift = 0.0407447

    @classmethod
    def set_filenames(cls):
        cls.bintemp = _get_bintemps(cls.dapver, default=True)
        cls.defaultbin, cls.defaulttemp = cls.bintemp.split('-', 1)
        cls.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu)
        cls.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu)
        cls.imgname = '{0}.png'.format(cls.ifu)
        cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
        cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)

    @classmethod
    def update_filenames(cls, bintype=None, template=None):
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


@pytest.fixtures(scope='session')
def marvinsession(request):
    return MarvinSession(request)


@pytest.fixtures(scope='session')
def galaxy(request):
    return Galaxy(request)


# TODO move to a mock data file/directory
galaxies = {'8485-1901': {'mangaid': '1-209232',
                          'cubepk': 10179,
                          'ra': 232.544703894,
                          'dec': 48.6902009334,
                          'redshift': 0.0407447}
            }


# --------- #
#           #
# Version 1 #
#           #
# --------- #

# TODO setupConfig, setupDB, setupPaths could be grouped into a larger FixtureManager class

@pytest.fixture(scope='session')
def setupConfig():
    config.use_sentry = False
    config.add_github_message = False


@pytest.fixture(scope='session')
def setupDB():
    # cls._marvindb = marvindb  # TODO fix instances of cls._marvindb
    # cls.session = marvindb.session  # TODO fix instances of cls.session
    session = marvindb.session


@pytest.fixture(scope='session')
def setupPaths():
    sasbasedir = os.getenv("$SAS_BASE_DIR")  # TODO fix instances of cls.sasbasedir
    mangaredux = os.getenv("MANGA_SPECTRO_REDUX")  # TODO fix instances of cls.mangaredux
    mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")  # TODO fix instances of cls.mangaanalysis


@pytest.fixture(scope='session')
def set_plateifu(plateifu='8485-1901'):
    plate, ifu = plateifu.split('-')
    plate = int(plate)
    return plate, ifu


galaxies = {'8485-1901': {'mangaid': '1-209232',
                          'cubepk': 10179,
                          'ra': 232.544703894,
                          'dec': 48.6902009334,
                          'redshift': 0.0407447}
            }

@pytest.fixture(scope='session')
def setupRelease(release):  # 'MPL-5'
    drpver, dapver = config.lookUpVersions(release)
    return release, drpver, dapver


@pytest.fixture(scope='session')
def setupGalaxy(plateifu='8485-1901'):
    plate, ifu = set_plateifu(plateifu=plateifu)
    galaxy = galaxies['plateifu']
    mangaid = galaxy['mangaid']
    cubepk = galaxy['cubepk']
    ra = galaxy['ra']
    dec = galaxy['dec']
    redshift = galaxy['redshift']
    return


cls.dir3d = 'stack'

# TODO add filename setup session fixture
cls.bintemp = _get_bintemps(cls.dapver, default=True)
cls.defaultbin, cls.defaulttemp = cls.bintemp.split('-', 1)
cls.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu)
cls.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu)
cls.imgname = '{0}.png'.format(cls.ifu)
cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)


# --------- #
#           #
# Version 0 #
#           #
# --------- #


@pytest.fixture(scope='session')
class MarvinTest:
    """Custom class for Marvin-tools tests."""

    # TODO: Replace instance of skipTest and skipBrian with pytest's skipif
    #
    # def skipTest(self, test):
    #     """Issues a warning when we skip a test."""
    #     warnings.warn('Skipped test {0} because there is no DB connection.'
    #                   .format(test.__name__), MarvinSkippedTestWarning)
    #
    # def skipBrian(self, test):
    #     """Issues a warning when we skip a test."""
    #     warnings.warn('Skipped test {0} because there is no Brian.'
    #                   .format(test.__name__), MarvinSkippedTestWarning)

    # TODO how to pass in classmethod without subclassing?
    # Can I do this as a session level fixture? Probably
    @classmethod
    def setUpClass(cls):
        config.use_sentry = False
        config.add_github_message = False

        # TODO use monkeypatch to safely set these attributes
        #
        # set initial config variables
        # cls.init_mode = config.mode
        # cls.init_sasurl = config.sasurl
        # cls.init_urlmap = config.urlmap
        # cls.init_xyorig = config.xyorig
        # cls.init_traceback = config._traceback
        # cls.init_keys = ['mode', 'sasurl', 'urlmap', 'xyorig', 'traceback']

        # set db stuff
        cls._marvindb = marvindb
        cls.session = marvindb.session

        # set paths
        cls.sasbasedir = os.getenv("$SAS_BASE_DIR")
        cls.mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
        cls.mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")

        # TODO refactor into separate method (setup_8485_1901)
        #
        # testing data for 8485-1901
        cls.set_plateifu(plateifu='8485-1901')
        cls.mangaid = '1-209232'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.redshift = 0.0407447
        cls.dir3d = 'stack'
        # TODO add release setup session fixture
        cls.release = 'MPL-5'
        cls.drpver, cls.dapver = config.lookUpVersions(cls.release)
        # TODO add filename setup session fixture
        cls.bintemp = _get_bintemps(cls.dapver, default=True)
        cls.defaultbin, cls.defaulttemp = cls.bintemp.split('-', 1)
        cls.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(cls.plateifu)
        cls.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(cls.plateifu)
        cls.imgname = '{0}.png'.format(cls.ifu)
        cls.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)
        cls.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(cls.plateifu, cls.bintemp)

    # TODO replace with monkeypatch
    #
    # def _reset_the_config(self):
    #     keys = self.init_keys
    #     for key in keys:
    #         ikey = 'init_{0}'.format(key)
    #         if hasattr(self, ikey):
    #             k = '_{0}'.format(key) if 'traceback' in key else key
    #             config.__setattr__(k, self.__getattribute__(ikey))

    @classmethod
    def set_sasurl(cls, loc='local', port=5000):
        istest = True if loc == 'utah' else False
        config.switchSasUrl(loc, test=istest, port=port)
        response = Interaction('api/general/getroutemap', request_type='get')
        config.urlmap = response.getRouteMap()

    # TODO replace with session level fixture (not in MarvinTest class)
    @classmethod
    def _update_release(cls, release):
        config.setMPL(release)
        cls.drpver, cls.dapver = config.lookUpVersions(release=release)

    # TODO replace with session level fixture (not in MarvinTest class)
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

    # TODO replace with session level fixture (not in MarvinTest class)
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

    # TODO replace with session level fixture (not in MarvinTest class)
    @classmethod
    def set_plateifu(cls, plateifu='8485-1901'):
        cls.plateifu = plateifu
        cls.plate, cls.ifu = cls.plateifu.split('-')
        cls.plate = int(cls.plate)

