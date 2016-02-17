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


__all__ = ['MarvinError', 'MarvinUserWarning', 'MarvinSkippedTestWargning']


class MarvinError(Exception):
    pass


class MarvinWarning(Warning):
    """Base warning for Marvin."""
    pass


class MarvinUserWarning(UserWarning, MarvinWarning):
    """The primary warning class."""
    pass


class MarvinSkippedTestWargning(MarvinUserWarning):
    """A warning for when a test is skipped."""
    pass
