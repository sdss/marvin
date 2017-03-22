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


class MarvinSession:
    """Custom class for Marvin-tools tests."""

    # necessary?
    # def __init__(self, request):
    #     self.request = request

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

    def set_db(self):
        self._marvindb = marvindb
        self.session = marvindb.session

    # def set_paths(self):
        # self.sasbasedir = os.getenv("$SAS_BASE_DIR")
        # self.mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
        # self.mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")

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
        # self.release = 'MPL-5'
        # self.drpver, self.dapver = config.lookUpVersions(self.release)
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

    # TODO What is the best way to test multiple releases (i.e., parameterization)?
    # We need multiple parameterization with releases and galaxies
    def _update_release(self, release):
        config.setMPL(release)
        self.drpver, self.dapver = config.lookUpVersions(release=release)


class Galaxy:
    """An example galaxy for Marvin-tools testing."""

    sasbasedir = os.getenv("$SAS_BASE_DIR")
    mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
    mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")
    dir3d = 'stack'

    def __init__(self, plateifu, release):
        self.plateifu = plateifu
        self.plate, self.ifu = self.plateifu.split('-')
        self.plate = int(self.plate)
        self.release = release
        self.drpver, self.dapver = config.lookUpVersions(self.release)

    # TODO move to a mock data file/directory
    def get_data(self):
        self.galaxies = {'8485-1901': {'mangaid': '1-209232',
                                       'cubepk': 10179,
                                       'ra': 232.544703894,
                                       'dec': 48.6902009334,
                                       'redshift': 0.0407447}
                         }

    def set_galaxy_data(self):
        # Grab data from mock file
        data = self.galaxies['plateifu']

        self.mangaid = data['mangaid']
        self.cubepk = data['cubepk']
        self.ra = data['ra']
        self.dec = data['dec']
        self.redshift = data['redshift']

    def set_filenames(self, bintype=None, template=None):
        default_bintemp = _get_bintemps(self.dapver, default=True)
        default_bin, default_temp = default_bintemp.split('-', 1)

        self.bintype = bintype if bintype is not None else default_bin
        self.template = template if template is not None else default_temp
        self.bintemp = '{0}-{1}'.format(self.bintype, self.template)

        self.cubename = 'manga-{0}-LOGCUBE.fits.gz'.format(self.plateifu)
        self.rssname = 'manga-{0}-LOGRSS.fits.gz'.format(self.plateifu)
        self.imgname = '{0}.png'.format(self.ifu)
        self.mapsname = 'manga-{0}-MAPS-{1}.fits.gz'.format(self.plateifu, self.bintemp)
        self.modelname = 'manga-{0}-LOGCUBE-{1}.fits.gz'.format(self.plateifu, self.bintemp)

    def set_filepaths(self):
        # Paths
        self.drppath = os.path.join(self.mangaredux, self.drpver)
        self.dappath = os.path.join(self.mangaanalysis, self.drpver, self.dapver)
        self.imgpath = os.path.join(self.mangaredux, self.drpver, str(self.plate), self.dir3d,
                                    'images')

        # DRP filename paths
        self.cubepath = os.path.join(self.drppath, str(self.plate), self.dir3d, self.cubename)
        self.rsspath = os.path.join(self.drppath, str(self.plate), self.dir3d, self.rssname)

        # DAP filename paths
        self.analysispath = os.path.join(self.dappath, self.bintemp, str(self.plate), self.ifu)
        self.mapspath = os.path.join(self.analysispath, self.mapsname)
        self.modelpath = os.path.join(self.analysispath, self.modelname)


@pytest.fixtures(scope='session')
def start_marvin_session():
    # TODO how do I access MarvinSession classmethods since it's no longer subclassed from
    # can I access `ms`?
    with MarvinSession() as ms:
        ms.set_config()
        ms.set_db()
        yield ms


@pytest.fixtures(scope='session')
def galaxy(request, start_marvin_session):
    with Galaxy(request.param) as gal:
        gal.get_data()
        gal.set_galaxy_data()
        gal.set_filenames()
        gal.set_filepaths()
        yield gal
