#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-02-13
# @Filename: exceptions.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-12 14:23:09


from __future__ import division, print_function

import os
import pwd
import sys
import warnings

import marvin


__all__ = ['MarvinError', 'MarvinUserWarning', 'MarvinSkippedTestWarning',
           'MarvinNotImplemented', 'MarvinMissingDependency', 'MarvinDeprecationWarning']


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
            try:
                self.client.context.merge({'user': {'name': pwd.getpwuid(os.getuid())[0],
                                                    'system': '_'.join(os.uname())}})
            except (OSError, IOError) as ee:
                warnings.warn('cannot initiate Sentry error reporting: {0}.'.format(str(ee)),
                              UserWarning)
                self.client = None
            except Exception as ee:
                warnings.warn('cannot initiate Sentry error reporting: unknown error.',
                              UserWarning)
                self.client = None

        else:
            self.client = None

ms = MarvinSentry(version=marvin.__version__)


class MarvinError(Exception):
    ''' Main Marvin Error '''
    def __init__(self, message=None, ignore_git=None):

        from marvin import config
        message = 'Unknown Marvin Error' if not message else message

        if config.use_sentry is True and ms.client is not None:
            # Send error to Sentry
            exc = sys.exc_info()
            if exc[0] is not None:
                ms.client.captureException(exc_info=exc)
            else:
                ms.client.captureMessage(message)

        if message[-1] != '.':
            message += '.'

        # Add Github Issue URL to message or not
        if config.add_github_message is True and not ignore_git:
            giturl = 'https://github.com/sdss/marvin/issues/new'
            message = ('{0}\nYou can submit this error to Marvin GitHub Issues ({1}).\n'
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


class MarvinDeprecationError(MarvinError):
    """To be raised for a deprecated feature."""
    pass


class MarvinWarning(Warning):
    """Base warning for Marvin."""
    pass


class MarvinUserWarning(UserWarning, MarvinWarning):
    """The primary warning class."""
    pass


class MarvinPassiveAggressiveWarning(MarvinUserWarning):
    """The passive aggressive warning class."""

    def __init__(self, message=None):
        message = ("Well, I wouldn't do it like that, but if that's "
                   "how you want to do it, sure, go ahead.\n{0}".format(message or ''))
        super(MarvinPassiveAggressiveWarning, self).__init__(message)


class MarvinSkippedTestWarning(MarvinUserWarning):
    """A warning for when a test is skipped."""
    pass


class MarvinDeprecationWarning(MarvinUserWarning):
    """A warning for deprecated features."""
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
