#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os
import pathlib
import json
import yaml
import pytest
import copy
import itertools
from flask_jwt_extended import tokens

from brain import bconfig

from marvin import config, marvindb
from marvin.utils.datamodel.dap import datamodel
from marvin.api.api import Interaction
from marvin.tools import Maps, Cube, ModelCube
from marvin.tools.query import Query
from sdss_access.path import Path




# Global Parameters for FIXTURES
# ------------------------------


releases = ['DR17']

bintypes_accepted = {'DR17': ['HYB10', 'SPX', 'VOR10']}

templates_accepted = {'DR17': ['MILESHC-MASTARSSP']}


def populate_bintypes_templates(releases, onlybin=None):
    ''' Generates bintype and template dictionaries for each release '''
    bintypes = {release: [] for release in releases}
    templates = {release: [] for release in releases}
    for release in releases:
        bintemps = datamodel[release].get_bintemps()
        for bintemp in bintemps:
            bintype = bintemp.split('-')[0]
            template = '-'.join(bintemp.split('-')[1:])

            if onlybin and bintype != onlybin:
                continue

            if release in bintypes_accepted and bintype not in bintypes_accepted[release]:
                continue
            if release in templates_accepted and template not in templates_accepted[release]:
                continue
            if bintype not in bintypes[release]:
                bintypes[release].append(bintype)
            if template not in templates[release]:
                templates[release].append(template)
    return bintypes, templates


bintypes, templates = populate_bintypes_templates(releases, onlybin='HYB10')

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
    return request.param


with open(os.path.join(os.path.dirname(__file__), 'data/query_test_data.dat')) as f:
    query_data = yaml.load(f, Loader=yaml.FullLoader)


with open(os.path.join(os.path.dirname(__file__), 'data/galaxy_test_data.dat')) as f:
    galaxy_data = yaml.load(f, Loader=yaml.FullLoader)


modes = ['local', 'remote']     # to loop over modes (see mode fixture)
dbs = ['db', 'nodb']                    # to loop over dbs (see db fixture)
origins = ['file', 'db', 'api']         # to loop over data origins (see data_origin fixture)


def pytest_addoption(parser):
    """Add new options"""
    # control releases run
    parser.addoption('--local-only', action='store_true', default=False, help='Run a local file tests only')


@pytest.fixture(scope='session')
def check_marks(pytestconfig):
    markers_arg = pytestconfig.getoption('-m')
    local = pytestconfig.getoption('--local-only')
    return markers_arg, local


@pytest.fixture(scope='session', params=sorted(galaxy_data.keys()))
def plateifu(request):
    """Yield a plate-ifu."""
    return request.param


@pytest.fixture(scope='session', params=origins)
def data_origin(request, check_marks):
    """Yield a data access mode."""
    marker, local = check_marks
    if local and request.param != 'file':
        pytest.skip('Skipping non-local modes')
    if ((marker == 'not uses_db' and request.param == 'db') or
        (marker == 'uses_db' and request.param != 'db')):
        pytest.skip('Skipping database modes')
    if ((marker == 'not uses_web' and request.param == 'api')
        or (marker == 'uses_web' and request.param != 'api')):
        pytest.skip('Skipping web/api modes')

    return request.param


@pytest.fixture(scope='session', params=modes)
def mode(request, check_marks):
    """Yield a data mode."""
    marker, local = check_marks
    if local and request.param != 'local':
        pytest.skip('Skipping non-local modes')
    if ((marker == 'not uses_web' and request.param == 'remote')
        or (marker == 'uses_web' and request.param != 'remote')):
        pytest.skip('Skipping web/api modes')

    return request.param


# Config-based FIXTURES
# ----------------------
def read_urlmap():
    """ read a test urlmap """
    path = pathlib.Path(__file__).parent / 'data/urlmap.json'
    with open(path, 'r') as f:
        return json.load(f)


URLMAP = read_urlmap()


@pytest.fixture(autouse=True)
def setup_config():
    config.access = 'public'


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


def set_the_config(release):
    """Set config release without parametrizing.

    Using ``set_release`` combined with ``galaxy`` double parametrizes!"""
    config.access = 'public'
    config.setRelease(release)
    set_sasurl(loc='local')
    #config.login()
    config._traceback = None

def custom_login():
    config.token = tokens.encode_access_token('test', os.environ.get('MARVIN_SECRET'), 'HS256', False, True, 'user_claims', True, 'identity', 'user_claims')

def custom_auth(self, authtype=None):
    authtype = 'token'
    super(Interaction, self).setAuth(authtype=authtype)


def set_sasurl(loc='local', port=None):
    """Set the sasurl to local or test-utah, and regenerate the urlmap."""
    if not port:
        port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
    istest = loc == 'utah'
    config.switchSasUrl(loc, test=istest, port=port)


@pytest.fixture(autouse=True)
def mock_urlmap(monkeypatch, mocker):
    """ Mock the urlmap """
    monkeypatch.setattr(config, 'urlmap', URLMAP)
    mocker.patch('marvin.config', new=config)


@pytest.fixture(scope='session', autouse=True)
def saslocal():
    """Set sasurl to local."""
    set_sasurl(loc='local')


@pytest.fixture(scope='session')
def set_release(release):
    """Set the release in the config."""
    config.setRelease(release)


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
    if not marvindb or not marvindb.isdbconnected:
        pytest.skip('Skipping when no database is connected')

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


@pytest.fixture()
def checkdb():
    """ Fixture to check if db available and turn off in marvin config """
    config.forceDbOn()
    nodb = not marvindb or not marvindb.isdbconnected
    if nodb:
        config.forceDbOff()
    yield
    config.forceDbOn()


@pytest.fixture(params=dbs)
def db(request, check_marks):
    """Turn local db on or off.

    Use this to parametrize over all db options.
    """
    if request.param == 'db' and (not marvindb or not marvindb.isdbconnected):
        pytest.skip('Skipping when no database is connected')

    marker, local = check_marks
    if local and request.param != 'nodb':
        pytest.skip('Skipping non-local db modes')
    if ((marker == 'not uses_db' and request.param == 'db') or
        (marker == 'uses_db' and request.param != 'notdb')):
        pytest.skip('Skipping database modes')

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
def monkeyauth(monkeypatch):
    monkeypatch.setattr(config, 'login', custom_login)
    monkeypatch.setattr(Interaction, 'setAuth', custom_auth)
    monkeypatch.setattr(bconfig, '_public_api_url', config.sasurl)
    monkeypatch.setattr(bconfig, '_collab_api_url', config.sasurl)


# Temp Dir/File-based FIXTURES
# ----------------------------
@pytest.fixture(scope='session')
def temp_scratch(tmp_path_factory):
    """Create a temporary scratch space for reading/writing.

    Use for creating temp dirs and files.

    Example at line 208 in tools/test_query, line 254 in tools/test_results, and
    misc/test_marvin_pickle.
    """
    fn = tmp_path_factory.mktemp('scratch')
    yield fn
    fn = None


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
                              'daptype': self.bintemp, 'wave': 'LOG'}

    def set_filepaths(self, pathtype='full'):
        """Set the paths for cube, maps, etc."""
        self.path = Path(release=self.release, public=True)
        self.imgpath = self.path.__getattribute__(pathtype)('mangaimage', **self.access_kwargs)
        self.cubepath = self.path.__getattribute__(pathtype)('mangacube', **self.access_kwargs)
        self.rsspath = self.path.__getattribute__(pathtype)('mangarss', **self.access_kwargs)

        if self.release == 'MPL-4':
            self.mapspath = self.path.__getattribute__(pathtype)('mangamap', **self.access_kwargs)
            self.modelpath = None
        else:
            self.access_kwargs.pop('mode')
            self.mapspath = self.path.__getattribute__(pathtype)('mangadap', mode='MAPS',
                                                                 **self.access_kwargs)
            self.modelpath = self.path.__getattribute__(pathtype)('mangadap', mode='LOGCUBE',
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
            name = 'mangamap' if self.release == 'MPL-4' else 'mangadap'
        elif name == 'modelcube':
            access_copy['mode'] = 'LOGCUBE'
            name = None if self.release == 'MPL-4' else 'mangadap'

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

    # if str(galaxy.bintype) != 'HYB10':
    #     pytest.skip()

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


@pytest.fixture(scope='function')
def maps_release_only(galaxy, release):
    return Maps(filename=galaxy.mapspath, release=release)


@pytest.fixture(scope='function')
@pytest.mark.uses_db
def query(request, allow_dap, release, mode, db):
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