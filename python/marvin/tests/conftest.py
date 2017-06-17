#!/usr/bin/env python
# encoding: utf-8
#
# conftest.py
#
# Created by Brett Andrews on 20 Mar 2017.


import itertools
import os

import pytest

from marvin import config, marvindb
from marvin.api.api import Interaction
from marvin.tools.query import Query
from marvin.tools.maps import _get_bintemps, __BINTYPES_MPL4__, __TEMPLATES_KIN_MPL4__

from sdss_access.path import Path

from astropy.io.misc import yaml


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='Run slow tests.')


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption('--runslow'):
        pytest.skip('Requires --runslow option to run.')


# You don't need this function, tmpdir_factory handles all this for you.
# See temp_scratch fixture and its use
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

releases = ['MPL-5', 'MPL-4']

bintypes = {release: [] for release in releases}
templates = {release: [] for release in releases}
for release in releases:
    __, dapver = config.lookUpVersions(release)
    bintemps = _get_bintemps(dapver)
    for bintemp in bintemps:
        bintype = bintemp.split('-')[0]
        template = '-'.join(bintemp.split('-')[1:])
        if bintype not in bintypes[release]:
            bintypes[release].append(bintype)
        if template not in templates[release]:
            templates[release].append(template)


# Galaxy data is stored in a YAML file
galaxy_data = yaml.load(open(os.path.join(os.path.dirname(__file__), 'data/galaxy_test_data.dat')))


class Galaxy(object):
    """An example galaxy for Marvin-tools testing."""

    sasbasedir = os.getenv('SAS_BASE_DIR')
    mangaredux = os.getenv('MANGA_SPECTRO_REDUX')
    mangaanalysis = os.getenv('MANGA_SPECTRO_ANALYSIS')
    dir3d = 'stack'

    def __init__(self, plateifu):
        self.plateifu = plateifu
        self.plate, self.ifu = self.plateifu.split('-')
        self.plate = int(self.plate)

    def set_galaxy_data(self, data_origin=None):
        """Sets galaxy properties from the configuration file."""

        data = galaxy_data[self.plateifu]

        for key in data.keys():
            setattr(self, key, data[key])

    def set_params(self, bintype=None, template=None, release=None):
        """Sets bintype, template, etc."""

        self.release = release
        self.drpver, self.dapver = config.lookUpVersions(self.release)
        self.drpall = 'drpall-{0}.fits'.format(self.drpver)

        default_bintemp = _get_bintemps(self.dapver, default=True)
        default_bin, default_temp = default_bintemp.split('-', 1)

        self.bintype = bintype if bintype is not None else default_bin
        self.template = template if template is not None else default_temp

        self.bintemp = '{0}-{1}'.format(self.bintype, self.template)

    def set_filepaths(self):
        """Sets the paths for cube, maps, etc."""

        path = Path()

        # Paths
        self.imgpath = os.path.join(self.mangaredux, self.drpver, str(self.plate), self.dir3d,
                                    'images')

        self.cubepath = path.full('mangacube', plateifu=self.plateifu,
                                  drpver=self.drpver, plate=self.plate, ifu=self.ifu)

        self.rsspath = path.full('mangarss', drpver=self.drpver, plate=self.plate, ifu=self.ifu)

        dap_params = dict(drpver=self.drpver, dapver=self.dapver,
                          plate=self.plate, ifu=self.ifu, bintype=self.bintype)

        if self.release == 'MPL-4':
            niter = int('{0}{1}'.format(__TEMPLATES_KIN_MPL4__.index(self.template),
                                        __BINTYPES_MPL4__[self.bintype]))
            self.mapspath = path.full('mangamap', n=niter, **dap_params)
            self.modelpath = None
        else:
            daptype = '{0}-{1}'.format(self.bintype, self.template)
            self.mapspath = path.full('mangadap5', mode='MAPS', daptype=daptype, **dap_params)
            self.modelpath = path.full('mangadap5', mode='LOGCUBE', daptype=daptype, **dap_params)


class DB(object):

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


@pytest.fixture(scope='session', params=galaxy_data.keys())
def get_plateifu(request):
    return request.param


# @pytest.fixture(scope='session', params=bintypes[config.release])
# def get_bintype(request):
#     return request.param
#
#
# @pytest.fixture(scope='session', params=templates[config.release])
# def get_template(request):
#     return request.param


def _get_release_generator_chain():
    """Returns a generator for all valid combinations of (release, bintype, template)."""

    return itertools.chain(*[itertools.product([release], bintypes[release],
                                               templates[release]) for release in releases])


def _params_ids(fixture_value):
    return '-'.join(fixture_value)


@pytest.fixture(scope='session', params=_get_release_generator_chain(), ids=_params_ids)
def get_params(request):
    """Yields a tuple of (release, bintype, template)."""

    return request.param


@pytest.fixture(scope='session', params=['file', 'db', 'api'])
def data_origin(request):
    """Yields a data access mode."""

    return request.param


@pytest.fixture(scope='function')
def db_off():
    """Turns the DB off and tears down."""

    config.forceDbOff()
    yield
    config.forceDbOn()


@pytest.fixture(scope='session')
def galaxy(maindb, get_params, get_plateifu, set_sasurl):

    release, bintype, template = get_params

    gal = Galaxy(plateifu=get_plateifu)
    gal.set_galaxy_data()
    gal.set_params(bintype=bintype, template=template, release=release)
    gal.set_filepaths()

    yield gal


# Query and Results Fixtures (loops over all modes and db possibilities)

modes = ['local', 'remote', 'auto']
dbs = ['db', 'nodb']


@pytest.fixture(params=modes)
def mode(request):
    return request.param


@pytest.fixture(params=dbs)
def db(request):
    ''' db fixture to turn on and off a local db'''
    if request.param == 'db':
        config.forceDbOn()
    else:
        config.forceDbOff()
    return config.db is not None


@pytest.fixture()
def query(request, set_release, set_sasurl, mode, db):
    if mode == 'local' and not db:
        pytest.skip('cannot use queries in local mode without a db')
    searchfilter = request.param if hasattr(request, 'param') else None
    q = Query(searchfilter=searchfilter, mode=mode)
    yield q
    config.forceDbOn()
    q = None
