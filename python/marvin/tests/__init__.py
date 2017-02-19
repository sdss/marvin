#!/usr/bin/env python
# encoding: utf-8

from unittest import TestCase
import warnings
import os
from marvin import config
from marvin.core.exceptions import MarvinSkippedTestWarning
from functools import wraps


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


# Decorator to skip a test if the session is None (i.e., if there is no DB)
def skipIfNoDB(test):
    @wraps(test)
    def wrapper(self, *args, **kwargs):
        if not self.session:
            return self.skipTest(test)
        else:
            return test(self, *args, **kwargs)
    return wrapper


# Decorator to skip if not Brian
def skipIfNoBrian(test):
    @wraps(test)
    def wrapper(self, *args, **kwargs):
        if 'Brian' not in os.path.expanduser('~'):
            return self.skipTest(test)
        else:
            return test(self, *args, **kwargs)
    return wrapper


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

        # testing data
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.plateifu = '8485-1901'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.redshift = 0.0407447

    def _reset_the_config(self):
        keys = [k for k in self.__dict__.keys() if 'init' in k]
        for key in keys:
            k = key.split('_')[1]
            k = '_' + k if 'traceback' in k else k
            config.__setattr__(k, self.__getattribute__(key))

    def set_sasurl(self, loc='local'):
        istest = True if loc == 'utah' else False
        config.switchSasUrl(loc, test=istest)
