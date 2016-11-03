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


__all__ = ['MarvinError', 'MarvinUserWarning', 'MarvinSkippedTestWarning',
           'MarvinNotImplemented', 'MarvinMissingDependency']


class MarvinError(Exception):
    pass


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
