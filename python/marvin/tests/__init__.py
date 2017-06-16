#!/usr/bin/env python
# encoding: utf-8

import warnings
import os
import inspect
from functools import wraps
from contextlib import contextmanager

import pytest
from unittest import TestCase

from marvin import config, marvindb
from marvin.core.exceptions import MarvinSkippedTestWarning
from marvin.api.api import Interaction
from marvin.tools.maps import _get_bintemps


def use_bintypes(*bintypes):
    """Decorates test to run only for the given bintypes."""
    def check_bintype(f):
        @wraps(f)
        def decorated_function(self, *args, **kwargs):
            if kwargs['galaxy'].bintype not in bintypes:
                pytest.skip('Only use {}'.format(', '.join(bintypes)))
            return f(self, *args, **kwargs)
        return decorated_function
    return check_bintype


class UseBintypes:
    """Decorate all tests in a class to run only for the given bintypes."""
    def __init__(self, *args):
        self.args = args

    def __call__(self, decorated_class):
        for attr in inspect.getmembers(decorated_class, inspect.isfunction):
            # only decorate public functions
            if attr[0][0] != '_':
                setattr(decorated_class, attr[0],
                        use_bintypes(*self.args)(getattr(decorated_class, attr[0])))
        return decorated_class


def use_releases(*releases):
    """Decorates test to run only for the given releases."""
    def check_bintype(f):
        @wraps(f)
        def decorated_function(self, *args, **kwargs):
            if 'release' in kwargs.keys():
                release = kwargs['release']
            elif 'galaxy' in kwargs.keys():
                release = kwargs['galaxy'].release
            if release not in releases:
                pytest.skip('Only use {}'.format(', '.join(releases)))
            return f(self, *args, **kwargs)
        return decorated_function
    return check_bintype


class UseReleases:
    """Decorate all tests in a class to run only for the given MPLs."""
    def __init__(self, *args):
        self.args = args

    def __call__(self, decorated_class):
        for attr in inspect.getmembers(decorated_class, inspect.isfunction):
            # only decorate public functions
            if attr[0][0] != '_':
                setattr(decorated_class, attr[0],
                        use_releases(*self.args)(getattr(decorated_class, attr[0])))

        return decorated_class


# These decorators for functions and classes allow to skip or run tests only for galaxies
# that have certain bintypes, templates, or releases
def marvin_skip_if(**kfilter):
    """Decorates test to skip/include certain parameters.

    kfilter is expected to be a dictionary of test fixtures, indicating the
    tests to skip or inclde. For instance,
    ``@marvin_skip_if(galaxy={'bintype': ['STON']})`` will skip any test in
    which the ``galaxy`` fixture has the attribute ``galaxy.bintype='STON'``.
    Individual fixtures can override the global ``mode``. In this example
    ``@marvin_skip_if(galaxy={'release': ['MPL-5'], mode='include'},
    data_mode=['file'], mode='skip')`` only tests with
    ``galaxy.release='MPL-5'`` will run, but any test in which
    ``data_mode='file'`` will be skipped.

    """
    def check_params(ff):

        def _should_skip(tmp_mode, filter_values, fixture_value, prop_name):
            ll = ', '.join(filter_values)
            if tmp_mode == 'skip' and fixture_value in filter_values:
                return pytest.skip('Skipping {0}={1!r}'.format(prop_name, ll))
            elif tmp_mode != 'skip' and fixture_value not in filter_values:
                return pytest.skip('Skipping all {0} except {1!r}'.format(prop_name, ll))

        @wraps(ff)
        def decorated_function(self, *args, **kwargs):
            global_mode = kfilter.get('mode', 'skip')
            for fixture_name, fixture_attributes in kfilter.items():
                if fixture_name == 'mode':
                    continue
                if fixture_name not in kwargs:
                    continue
                if not isinstance(fixture_attributes, dict):
                    tmp_mode = global_mode
                    fixture_value = kwargs[fixture_name]
                    filter_values = fixture_attributes
                    prop_name = fixture_name
                    _should_skip(tmp_mode, filter_values, fixture_value, prop_name)
                else:
                    tmp_mode = fixture_attributes.get('mode', global_mode)
                    for fixture_attribute, filter_values in fixture_attributes.items():
                        if fixture_attribute == 'mode':
                            continue
                        fixture = kwargs[fixture_name]
                        if not hasattr(fixture, fixture_attribute):
                            continue
                        fixture_value = getattr(fixture, fixture_attribute)
                        prop_name = fixture_attribute
                        _should_skip(tmp_mode, filter_values, fixture_value, prop_name)
            return ff(self, *args, **kwargs)
        return decorated_function
    return check_params


class marvin_skip_if_class(object):
    """Decorate all tests in a class to run only for, or skip, certain parameters."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, decorated_class):
        for attr in inspect.getmembers(decorated_class, inspect.isfunction):
            # only decorate public functions
            if attr[0][0] != '_':
                setattr(decorated_class, attr[0],
                        marvin_skip_if(*self.args,
                                       **self.kwargs)(getattr(decorated_class, attr[0])))
        return decorated_class


# Decorator to skip a test if the session is None (i.e., if there is no DB)
def skipIfNoDB(test):
    @wraps(test)
    def wrapper(self, db, *args, **kwargs):
        if db.session is None:
            pytest.skip('Skip because no DB.')
        else:
            return test(self, db, *args, **kwargs)
    return wrapper


@contextmanager
def set_tmp_sasurl(tmp_sasurl):
    sasurl = config.sasurl
    yield
    config.sasurl = sasurl


@contextmanager
def set_tmp_mpl(tmp_mpl):
    mpl = config.release
    yield
    config.setMPL(mpl)


# Moved from init in utils/test

# Copied from http://bit.ly/20i4EHC

def template(args):

    def wrapper(func):
        func.template = args
        return func

    return wrapper


def method_partial(func, *parameters, **kparms):
    @wraps(func)
    def wrapped(self, *args, **kw):
        kw.update(kparms)
        return func(self, *(args + parameters), **kw)
    return wrapped


class TemplateTestCase(type):

    def __new__(cls, name, bases, attr):

        new_methods = {}

        for method_name in attr:
            if hasattr(attr[method_name], "template"):
                source = attr[method_name]
                source_name = method_name.lstrip("_")

                for test_name, args in source.template.items():
                    parg, kwargs = args

                    new_name = "%s_%s" % (source_name, test_name)
                    new_methods[new_name] = method_partial(source, *parg,
                                                           **kwargs)
                    new_methods[new_name].__name__ = new_name

        attr.update(new_methods)
        return type(name, bases, attr)


def Call(*args, **kwargs):
    return (args, kwargs)

# Copied from init in tools/test


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
    def set_sasurl(cls, loc='local', port=None):
        if not port:
            port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
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
