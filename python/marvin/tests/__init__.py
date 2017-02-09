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
