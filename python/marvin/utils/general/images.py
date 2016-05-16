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
    from sdss_access import RsyncAccess
except ImportError:
    RsyncAccess = None

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
    '''Decorator that checks if RsyncAccess has been imported '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not RsyncAccess:
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

    rsync_access.add('mangaimage', plate='*', drpver=drpver, ifu='*', dir3d='stack')
    rsync_access.add('mangaimage', plate='*', drpver=drpver, ifu='*', dir3d='mastar')
    rsync_access.set_stream()

    if random: rsync_access.shuffle()
    listofimages = rsync_access.get_urls(limit=num) if as_url else rsync_access.get_paths(limit=num)

    if download:
        rsync_access.commit()
    else:
        return listofimages


@checkPath
@setMode
def getImagesByPlate(plateid, download=False, mode=None):
    ''' Get all images belonging to a given plate ID
    '''
    # setup Rsync Access
    rsync_access = RsyncAccess(label='marvintest', verbose=True)
    try:
        rsync_access.remote()
    except Exception as e:
        raise MarvinError('sdss_access .netrc file not installed: {0}'.format(e))

    # setup marvin inputs
    drpver = marvin.config.drpver

    rsync_access.add('mangaimage', plate=plateid, drpver=drpver, ifu='*', dir3d='stack')
    rsync_access.add('mangaimage', plate=plateid, drpver=drpver, ifu='*', dir3d='mastar')
    rsync_access.set_stream()

    listofimages = rsync_access.get_urls(limit=num) if as_url else rsync_access.get_paths(limit=num)

    if download:
        rsync_access.commit()
    else:
        return listofimages


@checkPath
@setMode
def getImagesByList(inputlist, download=False, mode=None):
    ''' Get all images from a list of ids
    '''
    # inputids = [list of plateifus, list of mangaids]

    # convert mangaids into plateifus

    for plateifu in inputlist:
        plateid, ifu = plateifu.split('-')
        rsync_access.add('mangaimage', plate=plateid, drpver=drpver, ifu=ifu, dir3d='stack')
    rsync_access.set_stream()

    # if marvin.mode == 'local': asurl=True
    # if marvin.model == 'remote':
    #        depends
    # if marvin.local and tools mode : depends

    listofimages = rsync_access.get_urls(limit=num) if as_url else rsync_access.get_paths(limit=num)

    if download:
        rsync_access.commit()
    else:
        return listofimages


