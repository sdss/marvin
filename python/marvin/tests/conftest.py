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

# class MarvinTest
# TODO Replace skipTest and skipBrian with skipif
# TODO use monkeypatch to set initial config variables
# TODO replace _reset_the_config with monkeypatch
# TODO reimplement set_sasurl (use function-level fixture?)

releases = ['MPL-4', 'MPL-5']
plateifus = ['8485-1901']

bintypes = {}
for release in releases:
    __, dapver = config.lookUpVersions(release)
    bintypes[release] = [bintemp.split('-')[0] for bintemp in _get_bintemps(dapver)]


class Galaxy:
    """An example galaxy for Marvin-tools testing."""

    sasbasedir = os.getenv("$SAS_BASE_DIR")
    mangaredux = os.getenv("MANGA_SPECTRO_REDUX")
    mangaanalysis = os.getenv("MANGA_SPECTRO_ANALYSIS")
    dir3d = 'stack'

    def __init__(self, plateifu):
        self.plateifu = plateifu
        self.plate, self.ifu = self.plateifu.split('-')
        self.plate = int(self.plate)
        self.release = config.release
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
        data = self.galaxies[self.plateifu]

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


class DB:
    def __init__(self):
        self._marvindb = marvindb
        self.session = marvindb.session


@pytest.fixture(scope='session')
def set_config():
    config.use_sentry = False
    config.add_github_message = False


@pytest.fixture(scope='session')
def set_sasurl(loc='local', port=5000):
    istest = True if loc == 'utah' else False
    config.switchSasUrl(loc, test=istest, port=port)
    response = Interaction('api/general/getroutemap', request_type='get')
    config.urlmap = response.getRouteMap()


@pytest.fixture(scope='session', params=releases)
def set_release(request):
    config.setMPL(request.param)


@pytest.fixture(scope='session')
def start_marvin_session(set_config):
    yield DB()


@pytest.fixture(scope='session', params=plateifus)
def get_plateifu(request):
    return request.param


@pytest.fixture(scope='session', params=bintypes[config.release])
def get_bintype(request):
    return request.param


@pytest.fixture(scope='session')
def galaxy(start_marvin_session, set_release, get_plateifu, get_bintype):
    gal = Galaxy(plateifu=get_plateifu)
    gal.get_data()
    gal.set_galaxy_data()
    gal.set_filenames(bintype=get_bintype)
    gal.set_filepaths()
    gal.maps_filename = gal.mapspath
    gal.modelcube_filename = gal.modelpath
    yield gal
