#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-03-20
# @Filename: conftest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-07-21 21:51:06

import copy
import itertools
import os
import warnings
from collections import OrderedDict

import pytest
import yaml
from brain import bconfig
from flask_jwt_extended import tokens
from sdss_access.path import Path

from marvin import config, marvindb
from marvin.api.api import Interaction
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tools.query import Query
from marvin.utils.datamodel.dap import datamodel


warnings.simplefilter('always')


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
        self.new_releases = ['MPL-6']
        self.new_bintypes = ['SPX', 'HYB10']
        self.new_templates = ['GAU-MILESHC']
        self.new_dbs = ['nodb']
        self.new_origins = ['file', 'api']
        self.new_modes = ['local', 'remote', 'auto']


# Global Parameters for FIXTURES
# ------------------------------
#releases = ['MPL-6', 'MPL-5', 'MPL-4']  # to loop over releases (see release fixture)
releases = ['MPL-8']

bintypes_accepted = {'MPL-4': ['NONE', 'VOR10'],
                     'MPL-5': ['SPX', 'VOR10'],
                     'MPL-6': ['SPX', 'HYB10'],
                     'MPL-7': ['HYB10', 'VOR10'],
                     'MPL-8': ['HYB10', 'SPX']}

templates_accepted = {'MPL-4': ['MIUSCAT_THIN', 'MILES_THIN'],
                      'MPL-5': ['GAU-MILESHC'],
                      'MPL-6': ['GAU-MILESHC'],
                      'MPL-7': ['GAU-MILESHC'],
                      'MPL-8': ['MILESHC-MILESHC']}


def populate_bintypes_templates(releases):
    ''' Generates bintype and template dictionaries for each release '''
    bintypes = OrderedDict((release, []) for release in releases)
    templates = OrderedDict((release, []) for release in releases)
    for release in releases:
        bintemps = datamodel[release].get_bintemps()
        for bintemp in bintemps:
            bintype = bintemp.split('-')[0]
            template = '-'.join(bintemp.split('-')[1:])
            if release in bintypes_accepted and bintype not in bintypes_accepted[release]:
                continue
            if release in templates_accepted and template not in templates_accepted[release]:
                continue
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
with open(os.path.join(os.path.dirname(__file__), 'data/galaxy_test_data.dat')) as f:
    galaxy_data = yaml.load(f)
with open(os.path.join(os.path.dirname(__file__), 'data/query_test_data.dat')) as f:
    query_data = yaml.load(f)


@pytest.fixture(scope='session', params=releases)
def release(request):
    """Yield a release."""
    if travis and request.param not in travis.new_releases:
        pytest.skip('Skipping non-requested release')

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
        pytest.skip('Skipping non-requested bintype')

    if travis and template not in travis.new_templates:
        pytest.skip('Skipping non-requested template')

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


@pytest.fixture(scope='session', params=modes)
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

URLMAP = None


def set_sasurl(loc='local', port=None):
    """Set the sasurl to local or test-utah, and regenerate the urlmap."""
    if not port:
        port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
    istest = True if loc == 'utah' else False
    config.switchSasUrl(loc, test=istest, port=port)
    global URLMAP
    if not URLMAP:
        response = Interaction('/marvin/api/general/getroutemap', request_type='get', auth='netrc')
        config.urlmap = response.getRouteMap()
        URLMAP = config.urlmap


@pytest.fixture(scope='session', autouse=True)
def saslocal():
    """Set sasurl to local."""
    set_sasurl(loc='local')


@pytest.fixture(scope='session')
def urlmap(saslocal):
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
    config.access = 'collab'
    config.setRelease(release)
    set_sasurl(loc='local')
    config.login()
    config._traceback = None


def custom_login():
    config.token = tokens.encode_access_token('test', os.environ.get('MARVIN_SECRET'), 'HS256', False, True, 'user_claims', True, 'identity', 'user_claims')


def custom_auth(self, authtype=None):
    authtype = 'token'
    super(Interaction, self).setAuth(authtype=authtype)


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


@pytest.fixture()
def user(maindb):
    username = 'test'
    password = 'test'
    model = maindb.datadb.User
    user = maindb.session.query(model).filter(model.username == username).one_or_none()
    if not user:
        user = model(username=username, login_count=1)
        user.set_password(password)
        maindb.session.add(user)
    yield user
    maindb.session.delete(user)


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


@pytest.fixture()
def monkeyauth(monkeypatch):
    monkeypatch.setattr(config, 'login', custom_login)
    monkeypatch.setattr(Interaction, 'setAuth', custom_auth)
    monkeypatch.setattr(bconfig, '_public_api_url', config.sasurl)
    monkeypatch.setattr(bconfig, '_collab_api_url', config.sasurl)


# Temp Dir/File-based FIXTURES
# ----------------------------
@pytest.fixture(scope='function')
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

        data = copy.deepcopy(galaxy_data[self.plateifu])

        for key in data.keys():
            setattr(self, key, data[key])

        # sets specfic data per release
        releasedata = self.releasedata[self.release]
        for key in releasedata.keys():
            setattr(self, key, releasedata[key])

        # remap NSA drpall names for MPL-4 vs 5+
        drpcopy = self.nsa_data['drpall'].copy()
        for key, val in self.nsa_data['drpall'].items():
            if isinstance(val, list):
                newval, newkey = drpcopy.pop(key)
                if self.release == 'MPL-4':
                    drpcopy[newkey] = newval
                else:
                    drpcopy[key] = newval
        self.nsa_data['drpall'] = drpcopy

    def set_params(self, bintype=None, template=None, release=None):
        """Set bintype, template, etc."""

        self.release = release

        self.drpver, self.dapver = config.lookUpVersions(self.release)
        self.drpall = 'drpall-{0}.fits'.format(self.drpver)

        self.bintype = datamodel[self.dapver].get_bintype(bintype)
        self.template = datamodel[self.dapver].get_template(template)
        self.bintemp = '{0}-{1}'.format(self.bintype.name, self.template.name)

        if release == 'MPL-4':
            self.niter = int('{0}{1}'.format(self.template.n, self.bintype.n))
        else:
            self.niter = '*'

        self.access_kwargs = {'plate': self.plate, 'ifu': self.ifu, 'drpver': self.drpver,
                              'dapver': self.dapver, 'dir3d': self.dir3d, 'mpl': self.release,
                              'bintype': self.bintype.name, 'n': self.niter, 'mode': '*',
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
            self.access_kwargs.pop('mode')
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
def galaxy(monkeyauth, get_params, plateifu):
    """Yield an instance of a Galaxy object for use in tests."""
    release, bintype, template = get_params

    set_the_config(release)
    gal = Galaxy(plateifu=plateifu)
    gal.set_params(bintype=bintype, template=template, release=release)
    gal.set_filepaths()
    gal.set_galaxy_data()
    yield gal
    gal = None


@pytest.fixture(scope='function')
def cube(galaxy, exporigin, mode):
    ''' Yield a Marvin Cube based on the expected origin combo of (mode+db).
        Fixture tests 6 cube origins from (mode+db) combos [file, db and api]
    '''

    if str(galaxy.bintype) != 'SPX':
        pytest.skip()

    if exporigin == 'file':
        c = Cube(filename=galaxy.cubepath, release=galaxy.release, mode=mode)
    else:
        c = Cube(plateifu=galaxy.plateifu, release=galaxy.release, mode=mode)

    c.exporigin = exporigin
    c.initial_mode = mode

    yield c

    c = None


@pytest.fixture(scope='function')
def modelcube(galaxy, exporigin, mode):
    ''' Yield a Marvin ModelCube based on the expected origin combo of (mode+db).
        Fixture tests 6 modelcube origins from (mode+db) combos [file, db and api]
    '''
    if exporigin == 'file':
        mc = ModelCube(filename=galaxy.modelpath, release=galaxy.release, mode=mode, bintype=galaxy.bintype)
    else:
        mc = ModelCube(plateifu=galaxy.plateifu, release=galaxy.release, mode=mode, bintype=galaxy.bintype)
    mc.exporigin = exporigin
    mc.initial_mode = mode
    yield mc
    mc = None


@pytest.fixture(scope='function')
def maps(galaxy, exporigin, mode):
    ''' Yield a Marvin Maps based on the expected origin combo of (mode+db).
        Fixture tests 6 cube origins from (mode+db) combos [file, db and api]
    '''
    if exporigin == 'file':
        m = Maps(filename=galaxy.mapspath, release=galaxy.release, mode=mode, bintype=galaxy.bintype)
    else:
        m = Maps(plateifu=galaxy.plateifu, release=galaxy.release, mode=mode, bintype=galaxy.bintype)
    m.exporigin = exporigin
    yield m
    m = None


modes = ['local', 'remote', 'auto']     # to loop over modes (see mode fixture)
dbs = ['db', 'nodb']                    # to loop over dbs (see db fixture)
origins = ['file', 'db', 'api']         # to loop over data origins (see data_origin fixture)


@pytest.fixture(scope='class')
def maps_release_only(release):
    return Maps(plateifu='8485-1901', release=release)


@pytest.fixture(scope='function')
def query(request, allow_dap, monkeyauth, release, mode, db):
    ''' Yields a Query that loops over all modes and db options '''
    data = query_data[release]
    set_the_config(release)
    if mode == 'local' and not db:
        pytest.skip('cannot use queries in local mode without a db')
    searchfilter = request.param if hasattr(request, 'param') else None
    q = Query(search_filter=searchfilter, mode=mode, release=release)
    q.expdata = data
    if q.mode == 'remote':
        pytest.xfail('cannot control for DAP spaxel queries on server side; failing all remotes until then')
    yield q
    config.forceDbOn()
    q = None


# @pytest.fixture(autouse=True)
# def skipall():
#     pytest.skip('skipping everything')


