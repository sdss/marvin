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
from functools import wraps
import marvin
import warnings
from distutils.version import StrictVersion
from marvin.utils.general import parseIdentifier, mangaid2plateifu

try:
    from sdss_access import RsyncAccess, AccessError
except ImportError:
    Path = None
    RsyncAccess = None

__all__ = ['getRandomImages', 'getImagesByPlate', 'getImagesByList']


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
        assert kwargs['mode'] in ['auto', 'local', 'remote'], 'Mode must be either auto, local, or remote'
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


def getDir3d(inputid, mode=None):
    ''' Get the 3d redux Image directory from an input plate or plate-IFU '''

    idtype = parseIdentifier(inputid)
    if idtype == 'plate':
        plateid = inputid
    elif idtype == 'plateifu':
        plateid, __ = inputid.split('-')

    release = marvin.config.release
    drpver, __ = marvin.config.lookUpVersions(release=release)
    drpstrict = StrictVersion(drpver.strip('v').replace('_', '.'))
    verstrict = StrictVersion('1.5.4')

    if drpstrict >= verstrict:
        from marvin.tools.plate import Plate
        plate = Plate(plateid=plateid, nocubes=True, mode=mode)
        dir3d = plate.dir3d
    else:
        dir3d = 'stack'

    return dir3d


# General image utilities
@checkPath
@setMode
def getRandomImages(num=10, download=False, mode=None, as_url=None, verbose=None, release=None):
    ''' Get a list of N random images from SAS

    Retrieve a random set of images from either your local filesystem SAS
    or the Utah SAS.  Optionally can download the images by rsync using
    sdss_access.

    Parameters:
        num (int):
            The number of images to retrieve
        download (bool):
            Set to download the images from the SAS
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`.
            the cube exists.
        as_url (bool):
            Convert the list of images to use the SAS url
        verbose (bool):
            Turns on verbosity during rsync
        release (str):
            The release version of the images to return

    Returns:
        listofimages (list):
            The list of images

    '''
    release = release if release else marvin.config.release
    drpver, __ = marvin.config.lookUpVersions(release=release)
    rsync_access = RsyncAccess(label='marvin_getrandom', verbose=verbose)

    if mode == 'local':
        full = rsync_access.full('mangaimage', plate='*', drpver=drpver, ifu='*', dir3d='stack')
        listofimages = rsync_access.random('', full=full, num=16, refine='\d{4,5}.png', as_url=True)
        return listofimages
    elif mode == 'remote':
        rsync_access.remote()
        rsync_access.add('mangaimage', plate='*', drpver=drpver, ifu='*', dir3d='stack')
        try:
            rsync_access.set_stream()
        except AccessError as e:
            raise MarvinError('Error with sdss_access rsync.set_stream. AccessError: {0}'.format(e))

        # refine and randomize
        rsync_access.refine_task('\d{4,5}.png')
        rsync_access.shuffle()
        listofimages = rsync_access.get_urls(limit=num) if as_url else rsync_access.get_paths(limit=num)

        if download:
            rsync_access.commit()
        else:
            return listofimages


@checkPath
@setMode
def getImagesByPlate(plateid, download=False, mode=None, as_url=None, verbose=None, release=None):
    ''' Get all images belonging to a given plate ID

    Retrieve all images belonging to a given plate ID from either your local filesystem SAS
    or the Utah SAS.  Optionally can download the images by rsync using
    sdss_access.

    When as_url is False, both local and remote modes will allow you to access
    the full path to the images in your local SAS.  WHen as_url is True,
    local mode generates the Utah SAS url links, while remote mode generates the
    Utah SAS rsync links.

    Auto mode defaults to remote.

    Parameters:
        plateid (int):
            The plate ID to retrieve the images for.  Required.
        download (bool):
            Set to download the images from the SAS
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`.
            the cube exists.
        as_url (bool):
            Convert the list of images to use the SAS url (mode=local)
            or the SAS rsync url (mode=remote)
        verbose (bool):
            Turns on verbosity during rsync
        release (str):
            The release version of the images to return

    Returns:
        listofimages (list):
            The list of images

    '''

    assert str(plateid).isdigit(), 'Plateid must be a numeric integer value'

    # setup Rsync Access
    rsync_access = RsyncAccess(label='marvin_getplate', verbose=verbose)

    # setup marvin inputs
    release = release if release else marvin.config.release
    drpver, __ = marvin.config.lookUpVersions(release=release)
    dir3d = getDir3d(plateid, mode=mode)

    # if mode is auto, set it to remote:
    if mode == 'auto':
        warnings.warn('Mode is auto.  Defaulting to remote.  If you want to access your \
            local images, set the mode explicitly to local', MarvinUserWarning)
        mode = 'remote'

    if mode == 'local':
        full = rsync_access.full('mangaimage', plate=plateid, drpver=drpver, ifu='*', dir3d=dir3d)
        listofimages = rsync_access.expand('', full=full, as_url=as_url)

        # if download, issue warning that cannot do it
        if download:
            warnings.warn('Download not available when in local mode', MarvinUserWarning)

        return listofimages
    elif mode == 'remote':
        rsync_access.remote()
        rsync_access.add('mangaimage', plate=plateid, drpver=drpver, ifu='*', dir3d=dir3d)

        # set the stream
        try:
            rsync_access.set_stream()
        except AccessError as e:
            raise MarvinError('Error with sdss_access rsync.set_stream. AccessError: {0}'.format(e))

        # get the list
        listofimages = rsync_access.get_urls() if as_url else rsync_access.get_paths()

        if download:
            rsync_access.commit()
        else:
            return listofimages


@checkPath
@setMode
def getImagesByList(inputlist, download=False, mode=None, as_url=None, verbose=None, release=None):
    ''' Get all images from a list of ids

    Retrieve a list of images from either your local filesystem SAS
    or the Utah SAS.  Optionally can download the images by rsync using
    sdss_access.

    When as_url is False, both local and remote modes will allow you to access
    the full path to the images in your local SAS.  WHen as_url is True,
    local mode generates the Utah SAS url links, while remote mode generates the
    Utah SAS rsync links.

    Auto mode defaults to remote.

    Parameters:
        inputlist (list):
            A list of plate-ifus or mangaids for the images you want to retrieve. Required.
        download (bool):
            Set to download the images from the SAS.  Only works in remote mode.
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`.
            the cube exists.
        as_url (bool):
            Convert the list of images to use the SAS url (mode=local)
            or the SAS rsync url (mode=remote)
        verbose (bool):
            Turns on verbosity during rsync
        release (str):
            The release version of the images to return

    Returns:
        listofimages (list):
            The list of images you have requested

    '''
    # Check inputs
    assert type(inputlist) == list or type(inputlist) == np.ndarray, 'Input must be of type list or Numpy array'
    idtype = parseIdentifier(inputlist[0])
    assert idtype in ['plateifu', 'mangaid'], 'Input must be of type plate-ifu or mangaid'
    # mode is checked via decorator

    # convert mangaids into plateifus
    if idtype == 'mangaid':
        newlist = []
        for myid in inputlist:
            try:
                plateifu = mangaid2plateifu(myid)
            except MarvinError as e:
                plateifu = None
            newlist.append(plateifu)
        inputlist = newlist

    # setup Rsync Access
    release = release if release else marvin.config.release
    drpver, __ = marvin.config.lookUpVersions(release=release)
    rsync_access = RsyncAccess(label='marvin_getlist', verbose=verbose)

    # if mode is auto, set it to remote:
    if mode == 'auto':
        warnings.warn('Mode is auto.  Defaulting to remote.  If you want to access your \
            local images, set the mode explicitly to local', MarvinUserWarning)
        mode = 'remote'

    # do a local or remote thing
    if mode == 'local':
        # Get list of images
        listofimages = []
        for plateifu in inputlist:
            dir3d = getDir3d(plateifu, mode=mode)
            plateid, ifu = plateifu.split('-')
            if as_url:
                path = rsync_access.url('mangaimage', plate=plateid, drpver=drpver, ifu=ifu, dir3d=dir3d)
            else:
                path = rsync_access.full('mangaimage', plate=plateid, drpver=drpver, ifu=ifu, dir3d=dir3d)
            listofimages.append(path)

        # if download, issue warning that cannot do it
        if download:
            warnings.warn('Download not available when in local mode', MarvinUserWarning)

        return listofimages
    elif mode == 'remote':
        rsync_access.remote()
        # Add plateifus to stream
        for plateifu in inputlist:
            dir3d = getDir3d(plateifu, mode=mode)
            plateid, ifu = plateifu.split('-')
            rsync_access.add('mangaimage', plate=plateid, drpver=drpver, ifu=ifu, dir3d=dir3d)

        # set the stream
        try:
            rsync_access.set_stream()
        except AccessError as e:
            raise MarvinError('Error with sdss_access rsync.set_stream. AccessError: {0}'.format(e))

        # get the list
        listofimages = rsync_access.get_urls() if as_url else rsync_access.get_paths()
        if download:
            rsync_access.commit()
        else:
            return listofimages
