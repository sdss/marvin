#!/usr/bin/env python
# encoding: utf-8
#
# conftest.py
#
# Created by Brett Andrews on 20 Mar 2017.


import os

import pytest
import pandas as pd

from marvin import config, marvindb
from marvin.api.api import Interaction
from marvin.tools.maps import _get_bintemps


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='Run slow tests.')


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption('--runslow'):
        pytest.skip('Requires --runslow option to run.')


# You don't need this function, tmpdir_factory handles all this for you.  see temp_scratch fixture and its use
@pytest.fixture(scope='function')
def tmpfiles():
    files_created = []

    yield files_created

    for fp in files_created:
        if os.path.exists(fp):
            os.remove(fp)


# TODO Replace skipTest and skipBrian with skipif
# TODO use monkeypatch to set initial config variables
# TODO replace _reset_the_config with monkeypatch

releases = ['MPL-5']  # TODO add 'MPL-4'
plateifus = ['8485-1901']  # TODO add '7443-12701'

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
        self.drpall = 'drpall-{0}.fits'.format(self.drpver)

    # TODO move to a mock data file/directory
    # TODO make release specific mock data sets
    def get_data(self):
        self.galaxies = pd.DataFrame(
                 [['1-209232', 10179, 221394, 232.544703894, 48.6902009334, 0.0407447, 0.234907269477844],
                  ['12-98126', 1001, 341153, 230.50746239, 43.53234133, 0.020478, 0.046455454081]],
                 columns=['mangaid', 'cubepk', 'nsaid', 'ra', 'dec', 'redshift', 'nsa_sersic_flux_ivar0'],
                 index=['8485-1901', '7443-12701'])

    def set_galaxy_data(self):
        # Grab data from mock file
        data = self.galaxies.loc[self.plateifu]

        for key in data.keys():
            setattr(self, key, data[key])

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
        self.datadb = marvindb.datadb
        self.sampledb = marvindb.sampledb
        self.dapdb = marvindb.dapdb


@pytest.fixture(scope='session')
def set_config():
    config.use_sentry = False
    config.add_github_message = False


@pytest.fixture()
def check_config():
    ''' check the config to see if a db is on or not '''
    return config.db is None


@pytest.fixture(scope='session')
def set_sasurl(loc='local', port=None):
    if not port:
        port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
    istest = True if loc == 'utah' else False
    config.switchSasUrl(loc, test=istest, port=port)
    response = Interaction('api/general/getroutemap', request_type='get')
    config.urlmap = response.getRouteMap()


@pytest.fixture(scope='session')
def urlmap(set_sasurl):
    return config.urlmap


# use if you need a temporary directory and or file space (see TestQueryPickling for usage)
@pytest.fixture(scope='session')
def temp_scratch(tmpdir_factory):
    fn = tmpdir_factory.mktemp('scratch')
    return fn


@pytest.fixture(scope='session', params=releases)
def release(request):
    return request.param


@pytest.fixture(scope='session')
def set_release(release):
    config.setMPL(release)


@pytest.fixture(scope='session')
def get_versions(set_release):
    drpver, dapver = config.lookUpVersions(config.release)
    return config.release, drpver, dapver


# TODO there is a fixture in test_query_pytest.py called ``db``
@pytest.fixture(scope='session')
def maindb(set_config):
    yield DB()


@pytest.fixture(scope='session', params=plateifus)
def get_plateifu(request):
    return request.param


@pytest.fixture(scope='session', params=bintypes[config.release])
def get_bintype(request):
    return request.param


@pytest.fixture(scope='session')
def galaxy(maindb, set_release, get_plateifu, get_bintype):
    gal = Galaxy(plateifu=get_plateifu)
    gal.get_data()
    gal.set_galaxy_data()
    gal.set_filenames(bintype=get_bintype)
    gal.set_filepaths()
    yield gal
