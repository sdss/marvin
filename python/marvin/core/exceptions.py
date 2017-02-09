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

        from marvin import config
        message = 'Unknown Marvin Error' if not message else message

        if config.use_sentry is True:
            # Send error to Sentry
            exc = sys.exc_info()
            if exc[0] is not None:
                ms.client.captureException(exc_info=exc)
            else:
                ms.client.captureMessage(message)

        # Add Github Issue URL to message or not
        if config.add_github_message is True:
            giturl = 'https://github.com/sdss/marvin/issues/new'
            message = ('{0}.\nYou can submit this error to Marvin GitHub Issues ({1}).\n'
                       'Fill out a subject and some text describing the error that just occurred.\n'
                       'If able, copy and paste the full traceback information into the issue '
                       'as well.'.format(message, giturl))

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
    """ A Sentry Breadcrumb to help leave a trail to bugs """

    def __init__(self):
        try:
            from raven import breadcrumbs
        except ImportError as e:
            breadcrumbs = None

        if breadcrumbs:
            self.breadcrumbs = breadcrumbs
        else:
            self.breadcrumbs = None

    def drop(self, **kwargs):
        ''' Records a breadcrumb into Sentry

            Info:
            https://docs.sentry.io/clients/python/breadcrumbs/
            https://docs.sentry.io/clientdev/interfaces/breadcrumbs/

        Parameters:
            timestamp (ISO datetime string, or a Unix timestamp):
                A timestamp representing when the breadcrumb occurred.
                This can be either an ISO datetime string, or a Unix timestamp.
            type (str):
                The type of breadcrumb. The default type is default which indicates
                no specific handling. Other types are currently http for HTTP requests and
                navigation for navigation events.
            message (str):
                If a message is provided it’s rendered as text where whitespace is preserved.
                Very long text might be abbreviated in the UI.
            data (dict):
                Data associated with this breadcrumb. Contains a sub-object whose contents
                depend on the breadcrumb type. See descriptions of breadcrumb types below.
                Additional parameters that are unsupported by the type are rendered as a key/value table.
            category (str):
                Categories are dotted strings that indicate what the crumb is or where it comes from.
                Typically it’s a module name or a descriptive string. For instance ui.click could
                be used to indicate that a click happend in the UI or flask could be used to indicate
                that the event originated in the Flask framework.
            level (str):
                This defines the level of the event. If not provided it defaults to info
                which is the middle level. In the order of priority from highest to lowest
                the levels are critical, error, warning, info and debug. Levels are
                used in the UI to emphasize and deemphasize the crumb.

        '''

        self.breadcrumbs.record(**kwargs)

