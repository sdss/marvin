#!/usr/bin/env python
# encoding: utf-8
#
# conftest.py
#
# Created by Brett Andrews on 20 Mar 2017.


from collections import OrderedDict
import itertools
import os

import pytest

from marvin import config, marvindb
from marvin.api.api import Interaction
from marvin.tools.cube import Cube
from marvin.tools.modelcube import ModelCube
from marvin.tools.maps import Maps
from marvin.tools.query import Query
from marvin.tools.maps import _get_bintemps, __BINTYPES_MPL4__, __TEMPLATES_KIN_MPL4__
from sdss_access.path import Path
import yaml


# PYTEST MODIFIERS
# -----------------
def pytest_addoption(parser):
    """Add new options"""
    # run slow tests
    parser.addoption('--runslow', action='store_true', default=False, help='Run slow tests.')
    # control releases run
    parser.addoption('--travis-only', action='store_true', default=False, help='Run a Travis only subset')


def pytest_runtest_setup(item):
    """Skip slow tests."""
    if 'slow' in item.keywords and not item.config.getoption('--runslow'):
        pytest.skip('Requires --runslow option to run.')


def pytest_configure(config):
    ''' Runs during configuration of conftest.  Checks and sets a global instance for a
        TravisSubset based on the pytest command line input of --travis-only
    '''
    option = config.getoption('--travis-only')
    global travis
    if option:
        travis = TravisSubset()

# specific release instance
travis = None


class TravisSubset(object):
    def __init__(self):
        self.new_gals = ['8485-1901']
        self.new_releases = ['MPL-5']
        self.new_bintypes = ['SPX']  # ['SPX', 'VOR10', 'NONE', 'STON']
        self.new_templates = ['GAU-MILESHC', 'MILES-THIN']
        self.new_dbs = ['nodb']
        self.new_origins = ['file', 'api']
        self.new_modes = ['local', 'remote']


# Global Parameters for FIXTURES
# ------------------------------
releases = ['MPL-5', 'MPL-4']           # to loop over releases (see release fixture)


def populate_bintypes_templates(releases):
    ''' Generates bintype and template dictionaries for each release '''
    bintypes = OrderedDict((release, []) for release in releases)
    templates = OrderedDict((release, []) for release in releases)
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
    return bintypes, templates


bintypes, templates = populate_bintypes_templates(releases)

# TODO reduce modes to only local and remote
modes = ['local', 'remote', 'auto']     # to loop over modes (see mode fixture)
dbs = ['db', 'nodb']                    # to loop over dbs (see db fixture)
origins = ['file', 'db', 'api']         # to loop over data origins (see data_origin fixture)


# Galaxy and Query data is stored in a YAML file
galaxy_data = yaml.load(open(os.path.join(os.path.dirname(__file__), 'data/galaxy_test_data.dat')))
query_data = yaml.load(open(os.path.join(os.path.dirname(__file__), 'data/query_test_data.dat')))


@pytest.fixture(scope='session', params=releases)
def release(request):
    """Yield a release."""
    return request.param


def _get_release_generator_chain():
    """Return all valid combinations of (release, bintype, template)."""
    return itertools.chain(*[itertools.product([release], bintypes[release],
                                               templates[release]) for release in releases])


def _params_ids(fixture_value):
    """Return a test id for the release chain."""
    return '-'.join(fixture_value)


@pytest.fixture(scope='session', params=sorted(_get_release_generator_chain()), ids=_params_ids)
def get_params(request):
    """Yield a tuple of (release, bintype, template)."""

    # placeholder until the marvin_test_if decorator works in 2.7
    release, bintype, template = request.param
    if travis and release not in travis.new_releases:
        pytest.skip('Skipping non-requested release')

    if travis and bintype not in travis.new_bintypes:
        pytest.skip('Skipping non-requested release')

    if travis and template not in travis.new_templates:
        pytest.skip('Skipping non-requested release')

    return request.param


@pytest.fixture(scope='session', params=sorted(galaxy_data.keys()))
def plateifu(request):
    """Yield a plate-ifu."""
    if travis and request.param not in travis.new_gals:
        pytest.skip('Skipping non-requested galaxies')
    return request.param


@pytest.fixture(scope='session', params=origins)
def data_origin(request):
    """Yield a data access mode."""
    if travis and request.param not in travis.new_origins:
        pytest.skip('Skipping non-requested origins')
    return request.param


@pytest.fixture(params=modes)
def mode(request):
    """Yield a data mode."""
    if travis and request.param not in travis.new_modes:
        pytest.skip('Skipping non-requested modes')
    return request.param


# Config-based FIXTURES
# ----------------------
@pytest.fixture(scope='session', autouse=True)
def set_config():
    """Set config."""
    config.use_sentry = False
    config.add_github_message = False
    config._traceback = None


@pytest.fixture()
def check_config():
    """Check the config to see if a db is on."""
    return config.db is None


@pytest.fixture(scope='session')
def set_sasurl(loc='local', port=None):
    """Set the sasurl to local or test-utah, and regenerate the urlmap."""
    if not port:
        port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
    istest = True if loc == 'utah' else False
    config.switchSasUrl(loc, test=istest, port=port)
    response = Interaction('api/general/getroutemap', request_type='get')
    config.urlmap = response.getRouteMap()


@pytest.fixture(scope='session', autouse=True)
def saslocal():
    """Set sasurl to local."""
    set_sasurl(loc='local')


@pytest.fixture(scope='session')
def urlmap(set_sasurl):
    """Yield the config urlmap."""
    return config.urlmap


@pytest.fixture(scope='session')
def set_release(release):
    """Set the release in the config."""
    config.setMPL(release)


@pytest.fixture(scope='session')
def versions(release):
    """Yield the DRP and DAP versions for a release."""
    drpver, dapver = config.lookUpVersions(release)
    return drpver, dapver


@pytest.fixture(scope='session')
def drpver(versions):
    """Return DRP version."""
    drpver, __ = versions
    return drpver


@pytest.fixture(scope='session')
def dapver(versions):
    """Return DAP version."""
    __, dapver = versions
    return dapver


def set_the_config(release):
    """Set config release without parametrizing.

    Using ``set_release`` combined with ``galaxy`` double parametrizes!"""
    config.setRelease(release)
    config._traceback = None


# DB-based FIXTURES
# -----------------

class DB(object):
    """Object representing aspects of the marvin db.

    Useful for tests needing direct DB access.
    """

    def __init__(self):
        """Initialize with DBs."""
        self._marvindb = marvindb
        self.session = marvindb.session
        self.datadb = marvindb.datadb
        self.sampledb = marvindb.sampledb
        self.dapdb = marvindb.dapdb


@pytest.fixture(scope='session')
def maindb():
    """Yield an instance of the DB object."""
    yield DB()


@pytest.fixture(scope='function')
def db_off():
    """Turn the DB off for a test, and reset it after."""
    config.forceDbOff()
    yield
    config.forceDbOn()


@pytest.fixture(autouse=True)
def db_on():
    """Automatically turn on the DB at collection time."""
    config.forceDbOn()


@pytest.fixture()
def usedb(request):
    ''' fixture for optional turning off the db '''
    if request.param:
        config.forceDbOn()
    else:
        config.forceDbOff()
    return config.db is not None


@pytest.fixture(params=dbs)
def db(request):
    """Turn local db on or off.

    Use this to parametrize over all db options.
    """
    if travis and request.param not in travis.new_dbs:
        pytest.skip('Skipping non-requested dbs')
    if request.param == 'db':
        config.forceDbOn()
    else:
        config.forceDbOff()
    yield config.db is not None
    config.forceDbOn()


@pytest.fixture()
def exporigin(mode, db):
    """Return the expected origins for a given db/mode combo."""
    if mode == 'local' and not db:
        return 'file'
    elif mode == 'local' and db:
        return 'db'
    elif mode == 'remote' and not db:
        return 'api'
    elif mode == 'remote' and db:
        return 'api'
    elif mode == 'auto' and db:
        return 'db'
    elif mode == 'auto' and not db:
        return 'file'


@pytest.fixture()
def expmode(mode, db):
    ''' expected modes for a given db/mode combo '''
    if mode == 'local' and not db:
        return None
    elif mode == 'local' and db:
        return 'local'
    elif mode == 'remote' and not db:
        return 'remote'
    elif mode == 'remote' and db:
        return 'remote'
    elif mode == 'auto' and db:
        return 'local'
    elif mode == 'auto' and not db:
        return 'remote'


# Monkeypatch-based FIXTURES
# --------------------------
@pytest.fixture()
def monkeyconfig(request, monkeypatch):
    """Monkeypatch a variable on the Marvin config.

    Example at line 160 in utils/test_general.
    """
    name, value = request.param
    monkeypatch.setattr(config, name, value=value)


@pytest.fixture()
def monkeymanga(monkeypatch, temp_scratch):
    """Monkeypatch the environ to create a temp SAS dir for reading/writing/downloading.

    Example at line 141 in utils/test_images.
    """
    monkeypatch.setitem(os.environ, 'SAS_BASE_DIR', str(temp_scratch))
    monkeypatch.setitem(os.environ, 'MANGA_SPECTRO_REDUX',
                        str(temp_scratch.join('mangawork/manga/spectro/redux')))
    monkeypatch.setitem(os.environ, 'MANGA_SPECTRO_ANALYSIS',
                        str(temp_scratch.join('mangawork/manga/spectro/analysis')))


# Temp Dir/File-based FIXTURES
# ----------------------------
@pytest.fixture(scope='session')
def temp_scratch(tmpdir_factory):
    """Create a temporary scratch space for reading/writing.

    Use for creating temp dirs and files.

    Example at line 208 in tools/test_query, line 254 in tools/test_results, and
    misc/test_marvin_pickle.
    """
    fn = tmpdir_factory.mktemp('scratch')
    return fn


def tempafile(path, temp_scratch):
    """Return a pytest temporary file given the original file path.

    Example at line 141 in utils/test_images.
    """
    redux = os.getenv('MANGA_SPECTRO_REDUX')
    anal = os.getenv('MANGA_SPECTRO_ANALYSIS')
    endredux = path.partition(redux)[-1]
    endanal = path.partition(anal)[-1]
    end = (endredux or endanal)

    return temp_scratch.join(end)


# Object-based FIXTURES
# ---------------------

class Galaxy(object):
    """An example galaxy for Marvin-tools testing."""

    sasbasedir = os.getenv('SAS_BASE_DIR')
    mangaredux = os.getenv('MANGA_SPECTRO_REDUX')
    mangaanalysis = os.getenv('MANGA_SPECTRO_ANALYSIS')
    dir3d = 'stack'

    def __init__(self, plateifu):
        """Initialize plate and ifu."""
        self.plateifu = plateifu
        self.plate, self.ifu = self.plateifu.split('-')
        self.plate = int(self.plate)

    def set_galaxy_data(self, data_origin=None):
        """Set galaxy properties from the configuration file."""

        if self.plateifu not in galaxy_data:
            return

        data = galaxy_data[self.plateifu]

        for key in data.keys():
            setattr(self, key, data[key])

        # sets specfic data per release
        releasedata = self.releasedata[self.release]
        for key in releasedata.keys():
            setattr(self, key, releasedata[key])

    def set_params(self, bintype=None, template=None, release=None):
        """Set bintype, template, etc."""
        self.release = release
        self.drpver, self.dapver = config.lookUpVersions(self.release)
        self.drpall = 'drpall-{0}.fits'.format(self.drpver)

        default_bintemp = _get_bintemps(self.dapver, default=True)
        default_bin, default_temp = default_bintemp.split('-', 1)

        self.bintype = bintype if bintype is not None else default_bin
        self.template = template if template is not None else default_temp

        self.bintemp = '{0}-{1}'.format(self.bintype, self.template)

        if release == 'MPL-4':
            self.niter = int('{0}{1}'.format(__TEMPLATES_KIN_MPL4__.index(self.template),
                                             __BINTYPES_MPL4__[self.bintype]))
        else:
            self.niter = '*'

        self.access_kwargs = {'plate': self.plate, 'ifu': self.ifu, 'drpver': self.drpver,
                              'dapver': self.dapver, 'dir3d': self.dir3d, 'mpl': self.release,
                              'bintype': self.bintype, 'n': self.niter, 'mode': '*',
                              'daptype': self.bintemp}

    def set_filepaths(self, pathtype='full'):
        """Set the paths for cube, maps, etc."""
        self.path = Path()
        self.imgpath = self.path.__getattribute__(pathtype)('mangaimage', **self.access_kwargs)
        self.cubepath = self.path.__getattribute__(pathtype)('mangacube', **self.access_kwargs)
        self.rsspath = self.path.__getattribute__(pathtype)('mangarss', **self.access_kwargs)

        if self.release == 'MPL-4':
            self.mapspath = self.path.__getattribute__(pathtype)('mangamap', **self.access_kwargs)
            self.modelpath = None
        else:
            __ = self.access_kwargs.pop('mode')
            self.mapspath = self.path.__getattribute__(pathtype)('mangadap5', mode='MAPS',
                                                                 **self.access_kwargs)
            self.modelpath = self.path.__getattribute__(pathtype)('mangadap5', mode='LOGCUBE',
                                                                  **self.access_kwargs)

    def get_location(self, path):
        """Extract the location from the input path."""
        return self.path.location("", full=path)

    def partition_path(self, path):
        """Partition the path into non-redux/analysis parts."""
        endredux = path.partition(self.mangaredux)[-1]
        endanalysis = path.partition(self.mangaanalysis)[-1]
        end = (endredux or endanalysis)
        return end

    def new_path(self, name, newvar):
        ''' Sets a new path with the subsituted name '''
        access_copy = self.access_kwargs.copy()
        access_copy['mode'] = '*'
        access_copy.update(**newvar)

        if name == 'maps':
            access_copy['mode'] = 'MAPS'
            name = 'mangamap' if self.release == 'MPL-4' else 'mangadap5'
        elif name == 'modelcube':
            access_copy['mode'] = 'LOGCUBE'
            name = None if self.release == 'MPL-4' else 'mangadap5'

        path = self.path.full(name, **access_copy) if name else None
        return path


@pytest.fixture(scope='function')
def galaxy(get_params, plateifu):
    """Yield an instance of a Galaxy object for use in tests."""
    release, bintype, template = get_params

    set_the_config(release)
    gal = Galaxy(plateifu=plateifu)
    gal.set_params(bintype=bintype, template=template, release=release)
    gal.set_filepaths()
    gal.set_galaxy_data()

    yield gal
    gal = None


@pytest.fixture()
def cube(galaxy, exporigin, mode):
    ''' Yield a Marvin Cube based on the expected origin combo of (mode+db).
        Fixture tests 6 cube origins from (mode+db) combos [file, db and api]
    '''
    if exporigin == 'file':
        c = Cube(filename=galaxy.cubepath, release=galaxy.release, mode=mode)
    else:
        c = Cube(plateifu=galaxy.plateifu, release=galaxy.release, mode=mode)
    c.exporigin = exporigin
    yield c
    c = None


@pytest.fixture()
def modelcube(galaxy, exporigin, mode):
    ''' Yield a Marvin ModelCube based on the expected origin combo of (mode+db).
        Fixture tests 6 modelcube origins from (mode+db) combos [file, db and api]
    '''
    if exporigin == 'file':
        mc = ModelCube(filename=galaxy.modelpath, release=galaxy.release, mode=mode)
    else:
        mc = ModelCube(plateifu=galaxy.plateifu, release=galaxy.release, mode=mode)
    mc.exporigin = exporigin
    yield mc
    mc = None


@pytest.fixture()
def maps(galaxy, exporigin, mode):
    ''' Yield a Marvin Maps based on the expected origin combo of (mode+db).
        Fixture tests 6 cube origins from (mode+db) combos [file, db and api]
    '''
    if exporigin == 'file':
        m = Maps(filename=galaxy.mapspath, release=galaxy.release, mode=mode)
    else:
        m = Maps(plateifu=galaxy.plateifu, release=galaxy.release, mode=mode)
    m.exporigin = exporigin
    yield m
    m = None


@pytest.fixture()
def query(request, release, mode, db):
    ''' Yields a Query that loops over all modes and db options '''
    data = query_data[release]
    set_the_config(release)
    if mode == 'local' and not db:
        pytest.skip('cannot use queries in local mode without a db')
    searchfilter = request.param if hasattr(request, 'param') else None
    q = Query(searchfilter=searchfilter, mode=mode, release=release)
    q.expdata = data
    yield q
    config.forceDbOn()
    q = None


# @pytest.fixture(autouse=True)
# def skipall():
#     pytest.skip('skipping everything')


