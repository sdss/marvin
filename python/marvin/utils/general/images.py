#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-29 00:04:16
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-29 00:04:16 by Brian Cherinka
    Last Modified On: 2016-04-29 00:04:16 by Brian

'''
from __future__ import print_function
from __future__ import division
from marvin.core.exceptions import MarvinError, MarvinUserWarning
import numpy as np
import warnings
from functools import wraps
import marvin

try:
    from sdss_access.path import Path
except ImportError:
    Path = None

__all__ = ['getRandomImages']


# Decorators
def setMode(func):
    '''Decorator that sets the mode to either config.mode or input mode '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        mode = kwargs.get('mode', None)
        if mode:
            kwargs['mode'] = mode
        else:
            kwargs['mode'] = marvin.config.mode
        return func(*args, **kwargs)
    return wrapper


def checkPath(func):
    '''Decorator that checks if Path has been imported '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not Path:
            raise MarvinError('sdss_access is not installed')
        else:
            return func(*args, **kwargs)
    return wrapper


# General image utilities
@checkPath
@setMode
def getRandomImages(num=10, download=False, mode=None):
    ''' Get a list of N random images from SAS

    '''
    print('mode', mode)


@checkPath
@setMode
def getImagesByPlate(plateid, download=False, mode=None):
    ''' Get all images belonging to a given plate ID
    '''
    pass


@checkPath
@setMode
def getImagesByList(download=False, mode=None):
    ''' Get all images from a list of ids
    '''
    pass




