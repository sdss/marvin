#!/usr/bin/env python
# encoding: utf-8
"""

exceptions.py

Licensed under a 3-clause BSD license.

Revision history:
    13 Feb 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function

import os
import sys
import marvin

__all__ = ['MarvinError', 'MarvinUserWarning', 'MarvinSkippedTestWarning',
           'MarvinNotImplemented', 'MarvinMissingDependency']


class MarvinSentry(object):
    ''' Sets up the Sentry python client '''

    def __init__(self, version='Unknown'):
        try:
            from raven import Client
        except ImportError as e:
            Client = None

        if Client:
            os.environ['SENTRY_DSN'] = 'https://98bc7162624049ffa3d8d9911e373430:1a6b3217d10e4207908d8e8744145421@sentry.io/107924'
            self.client = Client(
                    dsn=os.environ.get('SENTRY_DSN'),
                    release=version,
                    site='Marvin',
                    environment=sys.version.rsplit('|', 1)[0],
                    processors=(
                            'raven.processors.SanitizePasswordsProcessor',
                        )
                )
            self.client.context.merge({'user': {'name': os.getlogin(), 'system': '_'.join(os.uname())}})
        else:
            self.client = None

ms = MarvinSentry(version=marvin.__version__)


class MarvinError(Exception):
    def __init__(self, message=None):

        message = 'Unknown Marvin Error' if not message else message
        exc = sys.exc_info()
        if exc[0] is not None:
            ms.client.captureException(exc_info=exc)
        else:
            ms.client.captureMessage(message)
        super(MarvinError, self).__init__(message)


class MarvinNotImplemented(MarvinError):
    """A Marvin exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(MarvinNotImplemented, self).__init__(message)


class MarvinMissingDependency(MarvinError):
    """A custom exception for missing dependencies."""
    pass


class MarvinWarning(Warning):
    """Base warning for Marvin."""
    pass


class MarvinUserWarning(UserWarning, MarvinWarning):
    """The primary warning class."""
    pass


class MarvinSkippedTestWarning(MarvinUserWarning):
    """A warning for when a test is skipped."""
    pass


class MarvinBreadCrumb(object):
    pass
