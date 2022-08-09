#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-11-01
# @Filename: general.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-08-29 15:58:00


from __future__ import absolute_import, division, print_function

import collections
import contextlib
import inspect
import os
import re
import sys
import warnings
from builtins import range
from collections import OrderedDict
from functools import wraps
#from pkg_resources import parse_version
from packaging.version import parse

import matplotlib.pyplot as plt
import numpy as np
import PIL
from astropy import table, wcs
from astropy.units.quantity import Quantity
from brain.core.exceptions import BrainError
from flask_jwt_extended import get_jwt_identity
from scipy.interpolate import griddata

import marvin
from marvin import log
from marvin.core.exceptions import MarvinError, MarvinUserWarning


try:
    from sdss_access import Access, AccessError
except ImportError:
    Access = None

try:
    from sdss_access.path import Path
except ImportError:
    Path = None

try:
    import pympler.summary
    import pympler.muppy
    import psutil
except ImportError:
    pympler = None
    psutil = None


# General utilities
__all__ = ('convertCoords', 'parseIdentifier', 'mangaid2plateifu', 'findClosestVector',
           'getWCSFromPng', 'convertImgCoords', 'getSpaxelXY',
           'downloadList', 'getSpaxel', 'get_drpall_row', 'getDefaultMapPath',
           'getDapRedux', 'get_nsa_data', '_check_file_parameters',
           'invalidArgs', 'missingArgs', 'getRequiredArgs', 'getKeywordArgs',
           'isCallableWithArgs', 'map_bins_to_column', '_sort_dir', 'get_drpall_path',
           'get_dapall_path', 'temp_setattr', 'map_dapall', 'turn_off_ion', 'memory_usage',
           'validate_jwt', 'target_status', 'target_is_observed', 'target_is_mastar',
           'get_plates', 'get_manga_image', 'check_versions', 'get_drpall_table',
           'get_dapall_table', 'get_drpall_file', 'get_dapall_file')

drpTable = {}
dapTable = {}


def validate_jwt(f):
    ''' Decorator to validate a JWT and User '''

    @wraps(f)
    def wrapper(*args, **kwargs):
        current_user = get_jwt_identity()

        if not current_user:
            raise MarvinError('Invalid user from API token!')
        else:
            marvin.config.access = 'collab'
        return f(*args, **kwargs)
    return wrapper


def getSpaxel(cube=True, maps=True, modelcube=True,
              x=None, y=None, ra=None, dec=None, xyorig=None, **kwargs):
    """Returns the |spaxel| matching certain coordinates.

    The coordinates of the spaxel to return can be input as ``x, y`` pixels
    relative to``xyorig`` in the cube, or as ``ra, dec`` celestial
    coordinates.

    This function is intended to be called by
    :func:`~marvin.tools.cube.Cube.getSpaxel` or
    :func:`~marvin.tools.maps.Maps.getSpaxel`, and provides shared code for
    both of them.

    Parameters:
        cube (:class:`~marvin.tools.cube.Cube` or None or bool)
            A :class:`~marvin.tools.cube.Cube` object with the DRP cube
            data from which the spaxel spectrum will be extracted. If None,
            the |spaxel| object(s) returned won't contain spectral information.
        maps (:class:`~marvin.tools.maps.Maps` or None or bool)
            As ``cube`` but for the :class:`~marvin.tools.maps.Maps`
            object representing the DAP maps entity. If None, the |spaxel|
            will be returned without DAP information.
        modelcube (:class:`~marvin.tools.modelcube.ModelCube` or None or bool)
            As ``cube`` but for the :class:`~marvin.tools.modelcube.ModelCube`
            object representing the DAP modelcube entity. If None, the |spaxel|
            will be returned without model information.
        x,y (int or array):
            The spaxel coordinates relative to ``xyorig``. If ``x`` is an
            array of coordinates, the size of ``x`` must much that of
            ``y``.
        ra,dec (float or array):
            The coordinates of the spaxel to return. The closest spaxel to
            those coordinates will be returned. If ``ra`` is an array of
            coordinates, the size of ``ra`` must much that of ``dec``.
        xyorig ({'center', 'lower'}):
            The reference point from which ``x`` and ``y`` are measured.
            Valid values are ``'center'``, for the centre of the
            spatial dimensions of the cube, or ``'lower'`` for the
            lower-left corner. This keyword is ignored if ``ra`` and
            ``dec`` are defined. ``xyorig`` defaults to
            ``marvin.config.xyorig.``
        kwargs (dict):
            Arguments to be passed to `~marvin.tools.spaxel.SpaxelBase`.

    Returns:
        spaxels (list):
            The |spaxel| objects for this cube/maps corresponding to the input
            coordinates. The length of the list is equal to the number
            of input coordinates.

    .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

    """

    # TODO: for now let's put these imports here, but we should fix the
    # circular imports soon.
    import marvin.tools.cube
    import marvin.tools.maps
    import marvin.tools.modelcube
    import marvin.tools.spaxel

    # Checks that the cube and maps data are correct
    assert cube or maps or modelcube, \
        'Either cube, maps, or modelcube needs to be specified.'

    assert isinstance(cube, (marvin.tools.cube.Cube, bool)), \
        'cube is not an instance of Cube or a boolean'

    assert isinstance(maps, (marvin.tools.maps.Maps, bool)), \
        'maps is not an instance of Maps or a boolean'

    assert isinstance(modelcube, (marvin.tools.modelcube.ModelCube, bool)), \
        'modelcube is not an instance of ModelCube or a boolean'

    # Checks that we have the correct set of inputs.
    if x is not None or y is not None:
        assert ra is None and dec is None, 'Either use (x, y) or (ra, dec)'
        assert x is not None and y is not None, 'Specify both x and y'

        inputMode = 'pix'
        isScalar = np.isscalar(x)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        coords = np.array([x, y], float).T

    elif ra is not None or dec is not None:
        assert x is None and y is None, 'Either use (x, y) or (ra, dec)'
        assert ra is not None and dec is not None, 'Specify both ra and dec'

        inputMode = 'sky'
        isScalar = np.isscalar(ra)
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        coords = np.array([ra, dec], float).T

    else:
        raise ValueError('You need to specify either (x, y) or (ra, dec)')

    if not xyorig:
        xyorig = marvin.config.xyorig

    if isinstance(maps, marvin.tools.maps.Maps):
        ww = maps.wcs if inputMode == 'sky' else None
        cube_shape = maps._shape
    elif isinstance(cube, marvin.tools.cube.Cube):
        ww = cube.wcs if inputMode == 'sky' else None
        cube_shape = cube._shape
    elif isinstance(modelcube, marvin.tools.modelcube.ModelCube):
        ww = modelcube.wcs if inputMode == 'sky' else None
        cube_shape = modelcube._shape

    iCube, jCube = zip(convertCoords(coords, wcs=ww, shape=cube_shape,
                                     mode=inputMode, xyorig=xyorig).T)

    _spaxels = []
    for ii in range(len(iCube[0])):
        _spaxels.append(
            marvin.tools.spaxel.Spaxel(jCube[0][ii], iCube[0][ii],
                                       cube=cube, maps=maps, modelcube=modelcube, **kwargs))

    if len(_spaxels) == 1 and isScalar:
        return _spaxels[0]
    else:
        return _spaxels


def convertCoords(coords, mode='sky', wcs=None, xyorig='center', shape=None):
    """Convert input coordinates to array indices.

    Converts input positions in x, y or RA, Dec coordinates to array indices
    (in Numpy style) or spaxel extraction. In case of pixel coordinates, the
    origin of reference (either the center of the cube or the lower left
    corner) can be specified via ``xyorig``.

    If ``shape`` is defined (mandatory for ``mode='pix'``, optional for
    ``mode='sky'``) and one or more of the resulting indices are outside the
    size of the input shape, an error is raised.

    This functions is mostly intended for internal use.

    Parameters:
        coords (array):
            The input coordinates, as an array of shape Nx2.
        mode ({'sky', 'pix'}:
            The type of input coordinates, either `'sky'` for celestial
            coordinates (in the format defined in the WCS header information),
            or `'pix'` for pixel coordinates.
        wcs (None or ``astropy.wcs.WCS`` object):
            If ``mode='sky'``, the WCS solution from which the cube coordinates
            can be derived.
        xyorig (str):
            If ``mode='pix'``, the reference point from which the coordinates
            are measured. Valid values are ``'center'``, for the centre of the
            spatial dimensions of the cube, or ``'lower'`` for the lower-left
            corner.
        shape (None or array):
            If ``mode='pix'``, the shape of the spatial dimensions of the cube,
            so that the central position can be calculated.

    Returns:
        result (Numpy array):
            An array with the same shape as ``coords``, containing the cube
            index positions for the input coordinates, in Numpy style (i.e.,
            the first element being the row and the second the column).

    """

    coords = np.atleast_2d(coords)
    assert coords.shape[1] == 2, 'coordinates must be an array Nx2'

    if mode == 'sky':
        assert wcs, 'if mode==sky, wcs must be defined.'
        coordsSpec = np.ones((coords.shape[0], 3), np.float32)
        coordsSpec[:, :-1] = coords
        cubeCoords = wcs.wcs_world2pix(coordsSpec, 0)
        cubeCoords = np.fliplr(np.array(np.round(cubeCoords[:, :-1]), int))

    elif mode in ['pix', 'pixel']:
        assert xyorig, 'if mode==pix, xyorig must be defined.'
        x = coords[:, 0]
        y = coords[:, 1]

        assert shape is not None, 'if mode==pix, shape must be defined.'
        shape = np.atleast_1d(shape)

        if xyorig == 'center':
            yMid, xMid = shape / 2.
            xCube = np.round(xMid + x)
            yCube = np.round(yMid + y)
        elif xyorig == 'lower':
            xCube = np.round(x)
            yCube = np.round(y)
        else:
            raise ValueError('xyorig must be center or lower.')

        cubeCoords = np.array([yCube, xCube], int).T

    else:
        raise ValueError('mode must be pix or sky.')

    if shape is not None:
        if ((cubeCoords < 0).any() or
                (cubeCoords[:, 0] > (shape[0] - 1)).any() or
                (cubeCoords[:, 1] > (shape[1] - 1)).any()):
            raise MarvinError('some indices are out of limits.'
                              '``xyorig`` is currently set to "{0}". '
                              'Try setting ``xyorig`` to "{1}".'
                              .format(xyorig, 'center' if xyorig == 'lower' else 'lower'))

    return cubeCoords


def mangaid2plateifu(mangaid, mode='auto', drpall=None, drpver=None):
    """Return the plate-ifu for a certain mangaid.

    Uses either the DB or the drpall file to determine the plate-ifu for
    a mangaid. If more than one plate-ifu are available for a certain ifu,
    and ``mode='drpall'``, the one with the higher SN2 (calculated as the sum
    of redSN2 and blueSN2) will be used. If ``mode='db'``, the most recent one
    will be used.

    Parameters:
        mangaid (str):
            The mangaid for which the plate-ifu will be returned.
        mode ({'auto', 'drpall', 'db', 'remote'}):
            If `'drpall'` or ``'db'``, the  drpall file or the local database,
            respectively, will be used. If ``'remote'``, a request to the API
            will be issued. If ``'auto'``, the local modes will be tried before
            the remote mode.
        drpall (str or None):
            The path to the drpall file to use. If None, the file in
            ``config.drpall`` will be used.
        drpver (str or None):
            The DRP version to use. If None, the one in ``config.drpver`` will
            be used. If ``drpall`` is defined, this value is ignored.

    Returns:
        plateifu (str):
            The plate-ifu string for the input ``mangaid``.

    """

    from marvin import config, marvindb
    from marvin.api.api import Interaction

    # The modes and order over which the auto mode will loop.
    autoModes = ['db', 'drpall', 'remote']

    assert mode in autoModes + ['auto'], 'mode={0} is not valid'.format(mode)

    config_drpver, __ = config.lookUpVersions()
    drpver = drpver if drpver else config_drpver
    drpall = drpall if drpall else config._getDrpAllPath(drpver=drpver)

    if mode == 'drpall':

        # Get the drpall table from cache or fresh
        drpall_table = get_drpall_table(drpver=drpver, drpall=drpall)

        mangaids = np.array([mm.strip() for mm in drpall_table['mangaid']])

        plateifus = drpall_table[np.where(mangaids == mangaid)]

        if len(plateifus) > 1:
            warnings.warn('more than one plate-ifu found for mangaid={0}. '
                          'Using the one with the highest SN2.'.format(mangaid),
                          MarvinUserWarning)
            plateifus = plateifus[
                [np.argmax(plateifus['bluesn2'] + plateifus['redsn2'])]]

        if len(plateifus) == 0:
            raise ValueError('no plate-ifus found for mangaid={0}'.format(mangaid))

        return plateifus['plateifu'][0]

    elif mode == 'db':

        if not marvindb.isdbconnected:
            raise MarvinError('no DB connection found')

        if not drpver:
            raise MarvinError('drpver not set.')

        cubes = marvindb.session.query(marvindb.datadb.Cube).join(
            marvindb.datadb.PipelineInfo, marvindb.datadb.PipelineVersion).filter(
                marvindb.datadb.Cube.mangaid == mangaid,
                marvindb.datadb.PipelineVersion.version == drpver).use_cache().all()

        if len(cubes) == 0:
            raise ValueError('no plate-ifus found for mangaid={0}'.format(mangaid))
        elif len(cubes) > 1:
            warnings.warn('more than one plate-ifu found for mangaid={0}. '
                          'Using a the one with the higest SN2'.format(mangaid),
                          MarvinUserWarning)
            total_sn2 = [float(cube.header['BLUESN2']) + float(cube.header['REDSN2'])
                         for cube in cubes]
            cube = cubes[np.argmax(total_sn2)]
        else:
            cube = cubes[0]

        return '{0}-{1}'.format(cube.plate, cube.ifu.name)

    elif mode == 'remote':

        try:
            url = marvin.config.urlmap['api']['mangaid2plateifu']['url']
            response = Interaction(url.format(mangaid=mangaid))
        except MarvinError as e:
            raise MarvinError('API call to mangaid2plateifu failed: {0}'.format(e))
        else:
            plateifu = response.getData(astype=str)

        if not plateifu:
            if 'error' in response.results and response.results['error']:
                raise MarvinError(response.results['error'])
            else:
                raise MarvinError('API call to mangaid2plateifu failed with error unknown.')

        return plateifu

    elif mode == 'auto':

        for mm in autoModes:
            try:
                plateifu = mangaid2plateifu(mangaid, mode=mm, drpver=drpver, drpall=drpall)
                return plateifu
            except:
                continue

        raise MarvinError(
            'mangaid2plateifu was not able to find a plate-ifu for '
            'mangaid={0} either local or remotely.'.format(mangaid))


def findClosestVector(point, arr_shape=None, pixel_shape=None, xyorig=None):
    """Find the closest array coordinates from pixel coordinates.

    Find the closest vector of array coordinates (x, y) from an input
    vector of pixel coordinates (x, y).

    Parameters:
        point : tuple
            Original point of interest in pixel units, order of (x,y)
        arr_shape : tuple
            Shape of data array in (x,y) order
        pixel_shape : tuple
            Shape of image in pixels in (x,y) order
        xyorig : str
            Indicates the origin point of coordinates.  Set to
            "relative" switches to an array coordinate system relative
            to galaxy center.  Default is absolute array coordinates
            (x=0, y=0) = upper left corner

    Returns:
        minind : tuple
            A tuple of array coordinates in x, y order
    """

    # set as numpy arrays
    arr_shape = np.array(arr_shape, dtype=int)
    pixel_shape = np.array(pixel_shape, dtype=int)

    # compute midpoints
    xmid, ymid = arr_shape / 2
    xpixmid, ypixmid = pixel_shape / 2

    # default absolute array coordinates
    xcoords = np.array([0, arr_shape[0]], dtype=int)
    ycoords = np.array([0, arr_shape[1]], dtype=int)

    # split x,y coords and pixel coords
    x1, x2 = xcoords
    y1, y2 = ycoords
    xpix, ypix = pixel_shape

    # build interpolates between array coordinates and pixel coordinates
    points = [[x1, y1], [x1, y2], [xmid, ymid], [x2, y1], [x2, y2]]
    values = [[0, ypix], [0, 0], [xpixmid, ypixmid], [xpix, ypix], [xpix, 0]]  # full image
    # values = [[xpixmid-xmid, ypixmid+ymid], [xpixmid-xmid, ypixmid-ymid], [xpixmid, ypixmid], [xpixmid+xmid, ypixmid+ymid], [xpixmid+xmid, ypixmid-ymid]]  # pixels based on arr_shape
    #values = [[xpixmid-x2, ypixmid+y2], [xpixmid-x2, ypixmid-y2], [xpixmid, ypixmid], [xpixmid+x2, ypixmid+y2], [xpixmid+x2, ypixmid-y2]]  # pixels based on arr_shape

    # make 2d array of array indices in absolute or (our) relative coordindates
    arrinds = np.mgrid[x1:x2, y1:y2].swapaxes(0, 2).swapaxes(0, 1)
    # interpolate a new 2d pixel coordinate array
    final = griddata(points, values, arrinds)

    # find minimum array vector closest to input coordinate point
    diff = np.abs(point - final)
    prod = diff[:, :, 0] * diff[:, :, 1]
    minind = np.unravel_index(prod.argmin(), arr_shape)

    # toggle relative array coordinates
    if xyorig in ['relative', 'center']:
        minind = np.array(minind, dtype=int)
        xmin = minind[0] - xmid
        ymin = ymid - minind[1]
        minind = (xmin, ymin)

    return minind


def getWCSFromPng(filename=None, image=None):
    """Extract any WCS info from the metadata of a PNG image.

    Extracts the WCS metadata info from the PNG optical
    image of the galaxy using PIL (Python Imaging Library).
    Converts it to an Astropy WCS object.

    Parameters:
        image (object):
            An existing PIL image object
        filename (str):
            The full path to the image

    Returns:
        pngwcs (WCS):
            an Astropy WCS object
    """

    assert any([image, filename]), 'Must provide either a PIL image object, or the full image filepath'

    pngwcs = None

    if filename and not image:
        try:
            image = PIL.Image.open(filename)
        except Exception as e:
            raise MarvinError('Cannot open image {0}: {1}'.format(filename, e))
        else:
            # Close the image
            image.close()

    # get metadata
    meta = image.info if image else None

    # parse the image metadata
    mywcs = {}
    if meta and 'WCSAXES' in meta.keys():
        for key, val in meta.items():
            try:
                val = float(val)
            except Exception as e:
                pass
            mywcs.update({key: val})

        tmp = mywcs.pop('WCSAXES')

    # Construct Astropy WCS
    if mywcs:
        pngwcs = wcs.WCS(mywcs)

    return pngwcs


def convertImgCoords(coords, image, to_pix=None, to_radec=None):
    """Transform the WCS info in an image.

    Convert image pixel coordinates to RA/Dec based on
    PNG image metadata or vice_versa

    Parameters:
        coords (tuple):
            The input coordindates to transform
        image (str):
            The full path to the image
        to_pix (bool):
            Set to convert to pixel coordinates
        to_radec (bool):
            Set to convert to RA/Dec coordinates

    Returns:
        newcoords (tuple):
            Tuple of either (x, y) pixel coordinates
            or (RA, Dec) coordinates
    """

    try:
        wcs = getWCSFromPng(image)
    except Exception as e:
        raise MarvinError('Cannot get wcs info from image {0}: {1}'.format(image, e))

    if to_radec:
        try:
            newcoords = wcs.all_pix2world([coords], 1)[0]
        except AttributeError as e:
            raise MarvinError('Cannot convert coords to RA/Dec.  No wcs! {0}'.format(e))
    if to_pix:
        try:
            newcoords = wcs.all_world2pix([coords], 1)[0]
        except AttributeError as e:
            raise MarvinError('Cannot convert coords to image pixels.  No wcs! {0}'.format(e))
    return newcoords


def parseIdentifier(galid):
    """Determine if a string input is a plate, plateifu, or manga-id.

    Parses a string object id and determines whether it is a
    plate ID, a plate-IFU, or MaNGA-ID designation.

    Parameters:
        galid (str):
            The string of an id name to parse

    Returns:
        idtype (str):
            String indicating either plate, plateifu, mangaid, or None
    """

    galid = str(galid)
    hasdash = '-' in galid

    if hasdash:
        galidsplit = galid.split('-')

        if int(galidsplit[0]) > 6500:
            idtype = 'plateifu'
        else:
            idtype = 'mangaid'
    else:
        # check for plate
        if galid.isdigit():
            if len(galid) > 3:
                idtype = 'plate'
            else:
                idtype = None
        else:
            idtype = None

    return idtype


def getSpaxelXY(cube, plateifu, x, y):
    """Get a spaxel from a cube in the DB.

    This function is mostly intended for internal use.

    Parameters:
        cube (SQLAlchemy object):
            The SQLAlchemy object representing the cube from which to extract
            the spaxel.
        plateifu (str):
            The corresponding plateifu of ``cube``.
        x,y (int):
            The coordinates of the spaxel in the database.

    Returns:
        spaxel (SQLAlchemy object):
            The SQLAlchemy spaxel with coordinates ``(x, y)``.

    """

    import sqlalchemy

    mdb = marvin.marvindb

    try:
        spaxel = mdb.session.query(mdb.datadb.Spaxel).filter_by(cube=cube, x=x, y=y).use_cache().one()
    except sqlalchemy.orm.exc.NoResultFound as e:
        raise MarvinError('Could not retrieve spaxel for plate-ifu {0} at position {1},{2}: No Results Found: {3}'.format(plateifu, x, y, e))
    except Exception as e:
        raise MarvinError('Could not retrieve cube for plate-ifu {0} at position {1},{2}: Unknown exception: {3}'.format(plateifu, x, y, e))

    return spaxel


def getDapRedux(release=None):
    """Retrieve SAS url link to the DAP redux directory.

    Parameters:
        release (str):
            The release version of the data to download.
            Defaults to Marvin config.release.

    Returns:
        dapredux (str):
            The full redux path to the DAP MAPS
    """

    if not Path:
        raise MarvinError('sdss_access is not installed')
    else:
        # is_public = 'DR' in release
        # path_release = release.lower() if is_public else None
        sdss_path = Path(release=release)

    release = release or marvin.config.release
    drpver, dapver = marvin.config.lookUpVersions(release=release)
    ## hack a url version of MANGA_SPECTRO_ANALYSIS
    #dapdefault = sdss_path.dir('mangadefault', drpver=drpver, dapver=dapver, plate=None, ifu=None)
    #dappath = dapdefault.rsplit('/', 2)[0]
    dappath = os.path.join(os.getenv("MANGA_SPECTRO_ANALYSIS"), drpver, dapver)
    dapredux = sdss_path.url('', full=dappath)
    return dapredux


def getDefaultMapPath(**kwargs):
    """Retrieve the default Maps path.

    Uses sdss_access Path to generate a url download link to the
    default MAPS file for a given MPL.

    Parameters:
        release (str):
            The release version of the data to download.
            Defaults to Marvin config.release.
        plate (int):
            The plate id
        ifu (int):
            The ifu number
        mode (str):
            The bintype of the default file to grab, i.e. MAPS or LOGCUBE. Defaults to MAPS
        daptype (str):
            The daptype of the default map to grab.  Defaults to SPX-GAU-MILESHC

    Returns:
        maplink (str):
            The sas url to download the default maps file
    """

    # Get kwargs
    release = kwargs.get('release', marvin.config.release)
    drpver, dapver = marvin.config.lookUpVersions(release=release)
    plate = kwargs.get('plate', None)
    ifu = kwargs.get('ifu', None)
    daptype = kwargs.get('daptype', 'SPX-GAU-MILESHC')
    mode = kwargs.get('mode', 'MAPS')
    assert mode in ['MAPS', 'LOGCUBE'], 'mode can either be MAPS or LOGCUBE'

    # get sdss_access Path
    if not Path:
        raise MarvinError('sdss_access is not installed')
    else:
        # is_public = 'DR' in release
        # path_release = release.lower() if is_public else None
        sdss_path = Path(release=release)

    # get the sdss_path name by MPL
    # TODO: this is likely to break in future MPL/DRs. Just a heads up.
    if '4' in release:
        name = 'mangadefault'
    else:
        name = 'mangadap'

    # construct the url link to default maps file
    maplink = sdss_path.url(name, drpver=drpver, dapver=dapver, mpl=release,
                            plate=plate, ifu=ifu, daptype=daptype, mode=mode)
    return maplink


def downloadList(inputlist, dltype='cube', **kwargs):
    """Download a list of MaNGA objects.

    Uses sdss_access to download a list of objects
    via rsync.  Places them in your local sas path mimicing
    the Utah SAS.

    i.e. $SAS_BASE_DIR/mangawork/manga/spectro/redux

    Can download cubes, rss files, maps, modelcubes, mastar cubes,
    png images, default maps, or the entire plate directory.
    dltype=`dap` is a special keyword that downloads all DAP files.  It sets binmode
    and daptype to '*'

    Parameters:
        inputlist (list):
            Required.  A list of objects to download.  Must be a list of plate IDs,
            plate-IFUs, or manga-ids
        dltype ({'cube', 'maps', 'modelcube', 'dap', image', 'rss', 'mastar', 'default', 'plate'}):
            Indicated type of object to download.  Can be any of
            plate, cube, image, mastar, rss, map, modelcube, or default (default map).
            If not specified, the dltype defaults to cube.
        release (str):
            The MPL/DR version of the data to download.
            Defaults to Marvin config.release.
        bintype (str):
            The bin type of the DAP maps to download. Defaults to *
        binmode (str):
            The bin mode of the DAP maps to download. Defaults to *
        n (int):
            The plan id number [1-12] of the DAP maps to download. Defaults to *
        daptype (str):
            The daptype of the default map to grab.  Defaults to *
        dir3d (str):
            The directory where the images are located.  Either 'stack' or 'mastar'. Defaults to *
        verbose (bool):
            Turns on verbosity during rsync
        limit (int):
            A limit to the number of items to download
        test (bool):
            If True, tests the download path construction but does not download

    Returns:
        If test=True, returns the list of full filepaths that will be downloaded
    """

    assert isinstance(inputlist, (list, np.ndarray)), 'inputlist must be a list or numpy array'

    # Get some possible keywords
    # Necessary rsync variables:
    #   drpver, plate, ifu, dir3d, [mpl, dapver, bintype, n, mode]
    verbose = kwargs.get('verbose', None)
    as_url = kwargs.get('as_url', None)
    release = kwargs.get('release', marvin.config.release)
    drpver, dapver = marvin.config.lookUpVersions(release=release)
    bintype = kwargs.get('bintype', '*')
    binmode = kwargs.get('binmode', None)
    daptype = kwargs.get('daptype', '*')
    dir3d = kwargs.get('dir3d', '*')
    n = kwargs.get('n', '*')
    limit = kwargs.get('limit', None)
    test = kwargs.get('test', None)
    wave = 'LOG'

    # check for sdss_access
    if not Access:
        raise MarvinError('sdss_access not installed.')

    # Assert correct dltype
    dltype = 'cube' if not dltype else dltype
    assert dltype in ['plate', 'cube', 'mastar', 'modelcube', 'dap', 'rss', 'maps', 'image',
                      'default'], ('dltype must be one of plate, cube, mastar, '
                                   'image, rss, maps, modelcube, dap, default')

    assert binmode in [None, '*', 'MAPS', 'LOGCUBE'], 'binmode can only be *, MAPS or LOGCUBE'

    # Assert correct dir3d
    if dir3d != '*':
        assert dir3d in ['stack', 'mastar'], 'dir3d must be either stack or mastar'

    # Parse and retrieve the input type and the download type
    idtype = parseIdentifier(inputlist[0])
    if not idtype:
        raise MarvinError('Input list must be a list of plates, plate-ifus, or mangaids')

    # Set download type
    if dltype == 'cube':
        name = 'mangacube'
    elif dltype == 'rss':
        name = 'mangarss'
    elif dltype == 'default':
        name = 'mangadefault'
    elif dltype == 'plate':
        name = 'mangaplate'
    elif dltype == 'maps':
        # needs to change to include DR
        if '4' in release:
            name = 'mangamap'
        else:
            name = 'mangadap'
            binmode = 'MAPS'
    elif dltype == 'modelcube':
        name = 'mangadap'
        binmode = 'LOGCUBE'
    elif dltype == 'dap':
        name = 'mangadap'
        binmode = '*'
        daptype = '*'
    elif dltype == 'mastar':
        name = 'mangamastar'
    elif dltype == 'image':
        name = 'mangaimage'

    # create rsync
    rsync_access = Access(label='marvin_download', verbose=verbose, release=release)
    rsync_access.remote()

    # Add objects
    for item in inputlist:
        if idtype == 'mangaid':
            try:
                plateifu = mangaid2plateifu(item)
            except MarvinError:
                plateifu = None
            else:
                plateid, ifu = plateifu.split('-')
        elif idtype == 'plateifu':
            plateid, ifu = item.split('-')
        elif idtype == 'plate':
            plateid = item
            ifu = '*'

        rsync_access.add(name, plate=plateid, drpver=drpver, ifu=ifu, dapver=dapver, dir3d=dir3d,
                         mpl=release, bintype=bintype, n=n, mode=binmode, daptype=daptype,
                         wave=wave)

    # set the stream
    try:
        rsync_access.set_stream()
    except AccessError as e:
        raise MarvinError('Error with sdss_access rsync.set_stream. AccessError: {0}'.format(e))

    # get the list and download
    listofitems = rsync_access.get_urls() if as_url else rsync_access.get_paths()

    # print download location
    item = listofitems[0] if listofitems else None
    if item:
        ver = dapver if dapver in item else drpver
        dlpath = item[:item.rfind(ver) + len(ver)]
        if verbose:
            print('Target download directory: {0}'.format(dlpath))

    if test:
        return listofitems
    else:
        rsync_access.commit(limit=limit)


def _get_summary_file(name, summary_path=None, drpver=None, dapver=None):
    ''' Check for/download the drpall or dapall file

    Checks for existence of a local summary file for the
    current release set.  If not found, uses sdss_access
    to download it.

    Parameters:
        name (str):
            The name of the summary file.  Either drpall or dapall
        summary_path (str):
            The local path to either the drpall or dapall file
        drpver (str):
            The DRP version
        dapver (str):
            The DAP version
    '''

    assert name in ['drpall', 'dapall'], 'name must be either drpall or dapall'
    from marvin import config

    # # check for public release
    # is_public = 'DR' in config.release
    # release = config.release.lower() if is_public else None

    # get drpver and dapver
    config_drpver, config_dapver = config.lookUpVersions(config.release)
    drpver = drpver if drpver else config_drpver
    dapver = dapver if dapver else config_dapver

    if name == 'drpall' and not summary_path:
        summary_path = get_drpall_path(drpver)
    elif name == 'dapall' and not summary_path:
        summary_path = get_dapall_path(drpver, dapver)

    if not os.path.isfile(summary_path):
        warnings.warn('{0} file not found. Downloading it.'.format(name), MarvinUserWarning)
        rsync = Access(label='get_summary_file', release=config.release)
        rsync.remote()
        rsync.add(name, drpver=drpver, dapver=dapver)
        try:
            rsync.set_stream()
        except Exception as e:
            raise MarvinError('Could not download the {4} file with sdss_access: '
                              '{0}\nTry manually downloading it for version ({1},{2}) and '
                              'placing it {3}'.format(e, drpver, dapver, summary_path, name))
        else:
            rsync.commit()


def get_drpall_file(drpver=None, drpall=None):
    ''' Check for/download the drpall file

    Checks for existence of a local drpall file for the
    current release set.  If not found, uses sdss_access
    to download it.

    Parameters:
        drpver (str):
            The DRP version
        drpall (str):
            The local path to either the drpall file
    '''
    _get_summary_file('drpall', summary_path=drpall, drpver=drpver)


def get_dapall_file(drpver=None, dapver=None, dapall=None):
    ''' Check for/download the dapall file

    Checks for existence of a local dapall file for the
    current release set.  If not found, uses sdss_access
    to download it.

    Parameters:
        drpver (str):
            The DRP version
        dapver (str):
            The DAP version
        dapall (str):
            The local path to either the dapall file
    '''
    _get_summary_file('dapall', summary_path=dapall, drpver=drpver, dapver=dapver)


def get_drpall_row(plateifu, drpver=None, drpall=None):
    """Returns a dictionary from drpall matching the plateifu."""

    # get the drpall table
    drpall_table = get_drpall_table(drpver=drpver, drpall=drpall, hdu='MANGA')
    in_table = plateifu in drpall_table['plateifu']
    # check the mastar extension
    if not in_table:
        drpall_table = get_drpall_table(drpver=drpver, drpall=drpall, hdu='MASTAR')
        in_table = plateifu in drpall_table['plateifu']
        if not in_table:
            raise ValueError('No results found for {0} in drpall table'.format(plateifu))

    row = drpall_table[drpall_table['plateifu'] == plateifu]

    return row[0]


def _db_row_to_dict(row, remove_columns=False):
    """Converts a DB object to a dictionary."""

    from sqlalchemy.inspection import inspect as sa_inspect
    from sqlalchemy.ext.hybrid import hybrid_property

    row_dict = collections.OrderedDict()

    columns = row.__table__.columns.keys()

    mapper = sa_inspect(row.__class__)
    for key, item in mapper.all_orm_descriptors.items():

        if isinstance(item, hybrid_property):
            columns.append(key)

    for col in columns:
        if remove_columns and col in remove_columns:
            continue
        row_dict[col] = getattr(row, col)

    return row_dict


def get_nsa_data(mangaid, source='nsa', mode='auto', drpver=None, drpall=None):
    """Returns a dictionary of NSA data from the DB or from the drpall file.

    Parameters:
        mangaid (str):
            The mangaid of the target for which the NSA information will be returned.
        source ({'nsa', 'drpall'}):
            The data source. If ``source='nsa'``, the full NSA catalogue from the DB will
            be used. If ``source='drpall'``, the subset of NSA columns included in the drpall
            file will be returned.
        mode ({'auto', 'local', 'remote'}):
            See :ref:`mode-decision-tree`.
        drpver (str or None):
            The version of the DRP to use, if ``source='drpall'``. If ``None``, uses the
            version set by ``marvin.config.release``.
        drpall (str or None):
            A path to the drpall file to use if ``source='drpall'``. If not defined, the
            default drpall file matching ``drpver`` will be used.

    Returns:
        nsa_data (dict):
            A dictionary containing the columns and values from the NSA catalogue for
            ``mangaid``.

    """

    from marvin import config, marvindb
    from .structs import DotableCaseInsensitive

    valid_modes = ['auto', 'local', 'remote']
    assert mode in valid_modes, 'mode must be one of {0}'.format(valid_modes)

    valid_sources = ['nsa', 'drpall']
    assert source in valid_sources, 'source must be one of {0}'.format(valid_sources)

    log.debug('get_nsa_data: getting NSA data for mangaid=%r with source=%r, mode=%r',
              mangaid, source, mode)

    if mode == 'auto':
        log.debug('get_nsa_data: running auto mode mode.')
        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source=source,
                                    drpver=drpver, drpall=drpall)
            return nsa_data
        except MarvinError as ee:
            log.debug('get_nsa_data: local mode failed with error %s', str(ee))
            try:
                nsa_data = get_nsa_data(mangaid, mode='remote', source=source,
                                        drpver=drpver, drpall=drpall)
                return nsa_data
            except MarvinError as ee:
                raise MarvinError('get_nsa_data: failed to get NSA data for mangaid=%r in '
                                  'auto mode with with error: %s', mangaid, str(ee))

    elif mode == 'local':

        if source == 'nsa':

            if config.db is not None:

                session = marvindb.session
                sampledb = marvindb.sampledb

                nsa_row = session.query(sampledb.NSA).join(sampledb.MangaTargetToNSA,
                                                           sampledb.MangaTarget).filter(
                    sampledb.MangaTarget.mangaid == mangaid).use_cache().all()

                if len(nsa_row) == 1:
                    return DotableCaseInsensitive(
                        _db_row_to_dict(nsa_row[0], remove_columns=['pk', 'catalogue_pk']))
                elif len(nsa_row) > 1:
                    warnings.warn('get_nsa_data: multiple NSA rows found for mangaid={0}. '
                                  'Using the first one.'.format(mangaid), MarvinUserWarning)
                    return DotableCaseInsensitive(
                        _db_row_to_dict(nsa_row[0], remove_columns=['pk', 'catalogue_pk']))
                elif len(nsa_row) == 0:
                    raise MarvinError('get_nsa_data: cannot find NSA row for mangaid={0}'
                                      .format(mangaid))

            else:

                raise MarvinError('get_nsa_data: cannot find a valid DB connection.')

        elif source == 'drpall':

            plateifu = mangaid2plateifu(mangaid, drpver=drpver, drpall=drpall, mode='drpall')
            log.debug('get_nsa_data: found plateifu=%r for mangaid=%r', plateifu, mangaid)

            drpall_row = get_drpall_row(plateifu, drpall=drpall, drpver=drpver)

            nsa_data = collections.OrderedDict()
            for col in drpall_row.colnames:
                if col.startswith('nsa_'):
                    value = drpall_row[col]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    else:
                        # In Astropy 2 the value would be an array of size 1
                        # but in Astropy 3 value is already an scalar and asscalar fails.
                        try:
                            value = np.asscalar(value)
                        except AttributeError:
                            pass
                    nsa_data[col[4:]] = value

            return DotableCaseInsensitive(nsa_data)

    elif mode == 'remote':

        from marvin.api.api import Interaction

        try:
            if source == 'nsa':
                request_name = 'nsa_full'
            else:
                request_name = 'nsa_drpall'
            url = marvin.config.urlmap['api'][request_name]['url']
            response = Interaction(url.format(mangaid=mangaid))
        except MarvinError as ee:
            raise MarvinError('API call to {0} failed: {1}'.format(request_name, str(ee)))
        else:
            if response.results['status'] == 1:
                return DotableCaseInsensitive(collections.OrderedDict(response.getData()))
            else:
                raise MarvinError('get_nsa_data: %s', response['error'])


def _check_file_parameters(obj1, obj2):
    for param in ['plateifu', 'mangaid', 'plate', '_release', 'drpver', 'dapver']:
        assert_msg = ('{0} is different between {1} {2}:\n {1}.{0}: {3} {2}.{0}:{4}'
                      .format(param, obj1.__repr__, obj2.__repr__, getattr(obj1, param),
                              getattr(obj2, param)))
        assert getattr(obj1, param) == getattr(obj2, param), assert_msg


def add_doc(value):
    """Wrap method to programatically add docstring."""
    def _doc(func):
        func.__doc__ = value
        return func
    return _doc


def use_inspect(func):
    ''' Inspect a function of arguments and keywords.

    Inspects a function or class method.  Uses a different inspect for Python 2 vs 3
    Only tested to work with args and defaults.  varargs (variable arguments)
    and varkw (keyword arguments) seem to always be empty.

    Parameters:
        func (func):
            The function or method to inspect

    Returns:
        A tuple of arguments, variable arguments, keywords, and default values

    '''
    pyver = sys.version_info.major
    if pyver == 2:
        args, varargs, varkw, defaults = inspect.getargspec(func)
    elif pyver == 3:
        sig = inspect.signature(func)
        args = []
        defaults = []
        varargs = varkw = None
        for par in sig.parameters.values():
            # most parameters seem to be of this kind
            if par.kind == par.POSITIONAL_OR_KEYWORD:
                args.append(par.name)
                # parameters with default of inspect empty are required
                if par.default != inspect._empty:
                    defaults.append(par.default)

    return args, varargs, varkw, defaults


def getRequiredArgs(func):
    ''' Gets the required arguments from a function or method

    Uses this difference between arguments and defaults to indicate
    required versus optional arguments

    Parameters:
        func (func):
            The function or method to inspect

    Returns:
        A list of required arguments

    Example:
        >>> import matplotlib.pyplot as plt
        >>> getRequiredArgs(plt.scatter)
        >>> ['x', 'y']

    '''
    args, varargs, varkw, defaults = use_inspect(func)
    if defaults:
        args = args[:-len(defaults)]
    return args


def getKeywordArgs(func):
    ''' Gets the keyword arguments from a function or method

    Parameters:
        func (func):
            The function or method to inspect

    Returns:
        A list of keyword arguments

    Example:
        >>> import matplotlib.pyplot as plt
        >>> getKeywordArgs(plt.scatter)
        >>> ['edgecolors', 'c', 'vmin', 'linewidths', 'marker', 's', 'cmap',
        >>>  'verts', 'vmax', 'alpha', 'hold', 'data', 'norm']

    '''
    args, varargs, varkw, defaults = use_inspect(func)
    req_args = getRequiredArgs(func)
    opt_args = list(set(args) - set(req_args))
    return opt_args


def missingArgs(func, argdict, arg_type='args'):
    ''' Return missing arguments from an input dictionary

    Parameters:
        func (func):
            The function or method to inspect
        argdict (dict):
            The argument dictionary to test against
        arg_type (str):
            The type of arguments to test.  Either (args|kwargs|req|opt). Default is required.

    Returns:
        A list of missing arguments

    Example:
        >>> import matplotlib.pyplot as plt
        >>> testdict = {'edgecolors': 'black', 'c': 'r', 'xlim': 5, 'xlabel': 9, 'ylabel': 'y', 'ylim': 6}
        >>> # test for missing required args
        >>> missginArgs(plt.scatter, testdict)
        >>> {'x', 'y'}
        >>> # test for missing optional args
        >>> missingArgs(plt.scatter, testdict, arg_type='opt')
        >>> ['vmin', 'linewidths', 'marker', 's', 'cmap', 'verts', 'vmax', 'alpha', 'hold', 'data', 'norm']

    '''
    assert arg_type in ['args', 'req', 'kwargs', 'opt'], 'arg_type must be one of (args|req|kwargs|opt)'
    if arg_type in ['args', 'req']:
        return set(getRequiredArgs(func)).difference(argdict)
    elif arg_type in ['kwargs', 'opt']:
        return set(getKeywordArgs(func)).difference(argdict)


def invalidArgs(func, argdict):
    ''' Return invalid arguments from an input dictionary

    Parameters:
        func (func):
            The function or method to inspect
        argdict (dict):
            The argument dictionary to test against

    Returns:
        A list of invalid arguments

    Example:
        >>> import matplotlib.pyplot as plt
        >>> testdict = {'edgecolors': 'black', 'c': 'r', 'xlim': 5, 'xlabel': 9, 'ylabel': 'y', 'ylim': 6}
        >>> # test for invalid args
        >>> invalidArgs(plt.scatter, testdict)
        >>>  {'xlabel', 'xlim', 'ylabel', 'ylim'}

    '''
    args, varargs, varkw, defaults = use_inspect(func)
    return set(argdict) - set(args)


def isCallableWithArgs(func, argdict, arg_type='opt', strict=False):
    ''' Test if the function is callable with the an input dictionary

    Parameters:
        func (func):
            The function or method to inspect
        argdict (dict):
            The argument dictionary to test against
        arg_type (str):
            The type of arguments to test.  Either (args|kwargs|req|opt). Default is required.
        strict (bool):
            If True, validates input dictionary against both missing and invalid keyword arguments. Default is False

    Returns:
        Boolean indicating whether the function is callable

    Example:
        >>> import matplotlib.pyplot as plt
        >>> testdict = {'edgecolors': 'black', 'c': 'r', 'xlim': 5, 'xlabel': 9, 'ylabel': 'y', 'ylim': 6}
        >>> # test for invalid args
        >>> isCallableWithArgs(plt.scatter, testdict)
        >>> False

    '''
    if strict:
        return not missingArgs(func, argdict, arg_type=arg_type) and not invalidArgs(func, argdict)
    else:
        return not invalidArgs(func, argdict)


def map_bins_to_column(column, indices):
    ''' Maps a dictionary of array indices to column data

    Takes a given data column and a dictionary of indices (see the indices key
    from output of the histgram data in :meth:`marvin.utils.plot.scatter.hist`),
    and produces a dictionary with the data values from column mapped in
    individual bins.

    Parameters:
        column (list):
            A column of data
        indices (dict):
            A dictionary of providing a list of array indices belonging to each
            bin in a histogram.

    Returns:
        A dictionary containing, for each binid, a list of column data in that bin.

    Example:
        >>>
        >>> # provide a list of data in each bin of an output histogram
        >>> x = np.random.random(10)*10
        >>> hdata = hist(x, bins=3, return_fig=False)
        >>> inds = hdata['indices']
        >>> pmap = map_bins_to_column(x, inds)
        >>> OrderedDict([(1,
        >>>   [2.5092488009906235,
        >>>    1.7494530589363955,
        >>>    2.5070840461208754,
        >>>    2.188355400587354,
        >>>    2.6987990403658992,
        >>>    1.6023553861428441]),
        >>>  (3, [7.9214280403215875, 7.488908995456573, 7.190598204420587]),
        >>>  (4, [8.533028236560906])])

    '''
    assert isinstance(indices, dict) is True, 'indices must be a dictionary of binids'
    assert len(column) == sum(map(len, indices.values())), 'input column and indices values must have same len'
    coldict = OrderedDict()
    colarr = np.array(column)
    for key, val in indices.items():
        coldict[key] = colarr[val].tolist()
    return coldict


def _sort_dir(instance, class_):
    """Sort `dir()` to return child class attributes and members first.

    Return the attributes and members of the child class, so that
    ipython tab completion lists those first.

    Parameters:
        instance: Instance of `class_` (usually self).
        class_: Class of `instance`.

    Returns:
        list: Child class attributes and members.
    """
    members_array = list(zip(*inspect.getmembers(np.ndarray)))[0]
    members_quantity = list(zip(*inspect.getmembers(Quantity)))[0]
    members_parents = members_array + members_quantity

    return_list = [it[0] for it in inspect.getmembers(class_) if it[0] not in members_parents]
    return_list += vars(instance).keys()
    return_list += ['value']
    return return_list


def _get_summary_path(name, drpver, dapver=None):
    ''' Return the path for either the DRP or DAP summary file

    Parameters:
        name (str):
            The name of the summary file, either drpall or dapall
        drpver (str):
            The DRP version
        dapver (str):
            The DAP version
    '''
    assert name in ['drpall', 'dapall'], 'name must be either drpall or dapall'
    release = marvin.config.lookUpRelease(drpver)
    # is_public = 'DR' in release
    # path_release = release.lower() if is_public else None
    path = Path(release=release)
    all_path = path.full(name, drpver=drpver, dapver=dapver)
    return all_path


def get_drpall_path(drpver):
    """Returns the path to the DRPall file for ``(drpver, dapver)``."""

    drpall_path = _get_summary_path('drpall', drpver=drpver)
    return drpall_path


def get_dapall_path(drpver, dapver):
    """Returns the path to the DAPall file for ``(drpver, dapver)``."""

    dapall_path = _get_summary_path('dapall', drpver, dapver)
    return dapall_path


@contextlib.contextmanager
def turn_off_ion(show_plot=True):
    ''' Turns off the Matplotlib plt interactive mode

    Context manager to temporarily disable the interactive
    Matplotlib plotting functionality.  Useful for only returning
    Figure and Axes objects

    Parameters:
        show_plot (bool):
            If True, turns off the plotting

    Example:
        >>>
        >>> with turn_off_ion(show_plot=False):
        >>>     do_some_stuff
        >>>

    '''

    plt_was_interactive = plt.isinteractive()
    if not show_plot and plt_was_interactive:
        plt.ioff()

    fignum_init = plt.get_fignums()

    yield plt

    if show_plot:
        plt.ioff()
        plt.show()
    else:
        for ii in plt.get_fignums():
            if ii not in fignum_init:
                plt.close(ii)

    # Restores original ion() status
    if plt_was_interactive and not plt.isinteractive():
        plt.ion()


@contextlib.contextmanager
def temp_setattr(ob, attrs, new_values):
    """ Temporarily set attributed on an object

    Temporarily set an attribute on an object for the duration of the
    context manager.

    Parameters:
        ob (object):
            A class instance to set attributes on
        attrs (str|list):
            A list of attribute names to replace
        new_values (list):
            A list of new values to set as new attribute.  If new_values is
            None, all attributes in attrs will be set to None.

    Example:
        >>> c = Cube(plateifu='8485-1901')
        >>> print('before', c.mangaid)
        >>> with temp_setattr(c, 'mangaid', None):
        >>>     # do stuff
        >>>     print('new', c.mangaid)
        >>> print('after' c.mangaid)
        >>>
        >>> # Output
        >>> before '1-209232'
        >>> new None
        >>> after '1-209232'
        >>>

    """

    # set up intial inputs
    attrs = attrs if isinstance(attrs, list) else [attrs]
    if new_values:
        new_values = new_values if isinstance(new_values, list) else [new_values]
    else:
        new_values = [new_values] * len(attrs)

    assert len(attrs) == len(new_values), 'attrs and new_values must have the same length'

    replaced = []
    old_values = []

    # grab the old values
    for i, attr in enumerate(attrs):
        new_value = new_values[i]

        replace = False
        old_value = None
        if hasattr(ob, attr):
            try:
                if attr in ob.__dict__:
                    replace = True
            except AttributeError:
                if attr in ob.__slots__:
                    replace = True
            if replace:
                old_value = getattr(ob, attr)
        replaced.append(replace)
        old_values.append(old_value)
        setattr(ob, attr, new_value)

    # yield
    yield replaced, old_values

    # replace the old values
    for i, attr in enumerate(attrs):
        if not replaced[i]:
            delattr(ob, attr)
        else:
            setattr(ob, attr, old_values[i])


def map_dapall(header, row):
    ''' Retrieves a dictionary of DAPall db column names

    For a given row in the DAPall file, returns a dictionary
    of corresponding DAPall database columns names with the
    appropriate values.

    Parameters:
        header (Astropy header):
            The primary header of the DAPall file
        row (recarray):
            A row of the DAPall binary table data

    Returns:
        A dictionary with db column names as keys and row data as values

    Example:
        >>> hdu = fits.open('dapall-v2_3_1-2.1.1.fits')
        >>> header = hdu[0].header
        >>> row = hdu[1].data[0]
        >>> dbdict = map_dapall(header, row)

    '''

    # get names from header
    emline_schannels = []
    emline_gchannels = []
    specindex_channels = []
    for key, val in header.items():
        if 'ELS' in key:
            emline_schannels.append(val.lower().replace('-', '_').replace('.', '_'))
        elif 'ELG' in key:
            emline_gchannels.append(val.lower().replace('-', '_').replace('.', '_'))
        elif re.search('SPI([0-9])', key):
            specindex_channels.append(val.lower().replace('-', '_').replace('.', '_'))

    # File column names
    names = row.array.names

    dbdict = {}
    for col in names:
        name = col.lower()
        shape = row[col].shape if hasattr(row[col], 'shape') else ()
        array = ''
        values = row[col]

        if len(shape) > 0:
            channels = shape[0]
            for i in range(channels):
                channame = emline_schannels[i] if 'emline_s' in name else \
                    emline_gchannels[i] if 'emline_g' in name else \
                    specindex_channels[i] if 'specindex' in name else i + 1
                colname = '{0}_{1}'.format(name, channame)
                dbdict[colname] = values[i]
        else:
            dbdict[name] = values

    return dbdict


def get_virtual_memory_usage_kb():
    """
    The process's current virtual memory size in Kb, as a float.

    Returns:
        A float of the virtual memory usage

    """

    assert psutil is not None, 'the psutil python package is required to run this function'

    return float(psutil.Process().memory_info().vms) / 1024.0


def memory_usage(where):
    """
    Print out a basic summary of memory usage.

    Parameters:
        where (str):
            A string description of where in the code you are summarizing memory usage
    """

    assert pympler is not None, 'the pympler python package is required to run this function'

    mem_summary = pympler.summary.summarize(pympler.muppy.get_objects())
    print("Memory summary: {0}".format(where))
    pympler.summary.print_(mem_summary, limit=2)
    print("VM: {0:.2f}Mb".format(get_virtual_memory_usage_kb() / 1024.0))


def target_status(mangaid, mode='auto', source='nsa', drpall=None, drpver=None):
    ''' Check the status of a MaNGA target

    Given a mangaid, checks the status of a target.  Will check if
    target exists in the NSA catalog (i.e. is a proper target) and checks if
    target has a corresponding plate-IFU designation (i.e. has been observed).

    Returns a string status indicating if a target has been observed, has not
    yet been observed, or is not a valid MaNGA target.

    Parameters:
        mangaid (str):
            The mangaid of the target to check for observed status
        mode ({'auto', 'drpall', 'db', 'remote', 'local'}):
            See mode in :func:`mangaid2plateifu` and :func:`get_nsa_data`.
        source ({'nsa', 'drpall'}):
            The NSA catalog data source. See source in :func:`get_nsa_data`.
        drpall (str or None):
            The drpall file to use.  See drpall in :func:`mangaid2plateifu` and :func:`get_nsa_data`.
        drpver (str or None):
            The DRP version to use.  See drpver in :func:`mangaid2plateifu` and :func:`get_nsa_data`.

    Returns:
        A status of "observed", "not yet observed", or "not valid target"

    '''

    # check for plateifu - target has been observed
    try:
        plateifu = mangaid2plateifu(mangaid, mode=mode, drpver=drpver, drpall=drpall)
    except (MarvinError, BrainError) as e:
        plateifu = None

    # check if target in NSA catalog - proper manga target
    try:
        nsa = get_nsa_data(mangaid, source=source, mode=mode, drpver=drpver, drpall=drpall)
    except (MarvinError, BrainError) as e:
        nsa = None

    # return observed boolean
    if plateifu and nsa:
        status = 'observed'
    elif not plateifu and nsa:
        status = 'not yet observed'
    elif not plateifu and not nsa:
        status = 'not valid target'
    elif plateifu and not nsa:
        log.debug('No NSA info retrieved.  Cannot determine status of target {0}'.format(mangaid))
        status = 'unknown'

    return status


def target_is_observed(mangaid, mode='auto', source='nsa', drpall=None, drpver=None):
    ''' Check if a MaNGA target has been observed or not

    See :func:`target_status` for full documentation.

    Returns:
        True if the target has been observed.

    '''
    # check the target status
    status = target_status(mangaid, source=source, mode=mode, drpver=drpver, drpall=drpall)
    return status == 'observed'


def target_is_mastar(plateifu, drpver=None, drpall=None):
    ''' Check if a target is bright-time MaStar target

    Uses the local drpall file to check if a plateifu is a MaStar target

    Parameters:
        plateifu (str):
            The plateifu of the target
        drpver (str):
            The drpver version to check against
        drpall (str):
            The drpall file path

    Returns:
        True if it is

    '''

    row = get_drpall_row(plateifu, drpver=drpver, drpall=drpall)
    return row['srvymode'] == 'APOGEE lead'


def get_drpall_table(drpver=None, drpall=None, hdu='MANGA'):
    ''' Gets the drpall table

    Gets the drpall table either from cache or loads it. For releases
    of MPL-8 and up, galaxies are in the MANGA extension, and mastar
    targets are in the MASTAR extension, specified with the hdu keyword. For
    MPLs 1-7, there is only one data extension, which is read.

    Parameters:
        drpver (str):
            The DRP release version to load.  Defaults to current marvin release
        drpall (str):
            The full path to the drpall table. Defaults to current marvin release.
        hdu (str):
            The name of the HDU to read in.  Default is 'MANGA'

    Returns:
        An Astropy Table
    '''

    from marvin import config
    assert hdu.lower() in ['manga', 'mastar'], 'hdu can either be MANGA or MASTAR'
    hdu = hdu.upper()

    # get the drpall file
    get_drpall_file(drpall=drpall, drpver=drpver)

    # Loads the drpall table if it was not cached from a previous session.
    config_drpver, __ = config.lookUpVersions()
    drpver = drpver if drpver else config_drpver

    # check for drpver
    if drpver not in drpTable:
        drpTable[drpver] = {}

    # check for hdu
    hduext = hdu if check_versions(drpver, 'v2_5_3') else 'MANGA'
    if hdu not in drpTable[drpver]:
        drpall = drpall if drpall else get_drpall_path(drpver=drpver)
        data = {hduext: table.Table.read(drpall, hdu=hduext)}
        drpTable[drpver].update(data)

    drpall_table = drpTable[drpver][hduext]

    return drpall_table


def get_dapall_table(drpver=None, dapver=None, dapall=None):
    ''' Gets the dapall table

    Gets the dapall table either from cache or loads it. For releases
    of MPL-6 and up.

    Parameters:
        drpver (str):
            The DRP release version to load.  Defaults to current marvin release
        dapall (str):
            The full path to the dapall table. Defaults to current marvin release.

    Returns:
        An Astropy Table
    '''

    from marvin import config

    # get the dapall file
    get_dapall_file(dapall=dapall, drpver=drpver, dapver=dapver)

    # Loads the dapall table if it was not cached from a previous session.
    config_drpver, config_dapver = config.lookUpVersions(config.release)
    drpver = drpver or config_drpver
    dapver = dapver or config_dapver

    # check for dapver
    if dapver not in dapTable:
        dapall = dapall or get_dapall_path(drpver=drpver, dapver=dapver)
        if check_versions(dapver, '3.1.0'):
            tables = [table.Table.read(dapall, hdu=i) for i in range(1, 5)]
            data = table.vstack(tables, metadata_conflicts='silent')
        else:
            data = table.Table.read(dapall, hdu=1)
        dapTable[dapver] = data

    return dapTable[dapver]


def get_plates(drpver=None, drpall=None, release=None):
    ''' Get a list of unique plates from the drpall file

    Parameters:
        drpver (str):
            The DRP release version to load.  Defaults to current marvin release
        drpall (str):
            The full path to the drpall table. Defaults to current marvin release.
        release (str):
            The marvin release

    Returns:
        A list of plate ids

    '''
    assert not all([drpver, release]), 'Cannot set both drpver and release '

    if release:
        drpver, __ = marvin.config.lookUpVersions(release)

    drpall_table = get_drpall_table(drpver=drpver, drpall=drpall)
    plates = list(set(drpall_table['plate']))
    return plates


def check_versions(version1, version2):
    ''' Compate two version ids against each other

    Checks if version1 is greater than or equal to version2.

    Parameters:
        version1 (str):
            The version to check
        version2 (str):
            The version to check against

    Returns:
        A boolean indicating if version1 is >= version2
    '''

    return parse(version1) >= parse(version2)


def get_manga_image(cube=None, drpver=None, plate=None, ifu=None, dir3d=None, local=None, public=None):
    ''' Get a MaNGA IFU optical PNG image

    Parameters:
        cube (Cube):
            A Marvin Cube instance
        drpver (str):
            The drpver version
        plate (str|int):
            The plate id
        ifu (str|int):
            An IFU designation
        dir3d (str):
            The directory for 3d data, either 'stack' or 'mastar'
        local (bool):
            If True, return the local file path to the image
        public (bool):
            If True, use only DR releases
    Returns:
        A url to an IFU MaNGA image
    '''

    # check inputs
    drpver = cube._drpver if cube else drpver
    plate = cube.plate if cube else plate
    ifu = cube.ifu if cube else ifu
    dir3d = cube.dir3d if cube else dir3d
    assert all([drpver, plate, ifu]), 'drpver, plate, and ifu must be specified'

    # create the sdss Path
    release = cube.release if cube else marvin.config.lookUpRelease(drpver, public_only=public)
    path = Path(release=release)

    dir3d = dir3d if dir3d else 'stack'
    assert dir3d in ['stack', 'mastar'], 'dir3d can only be stack or mastar'
    if local:
        img = path.full('mangaimage', drpver=drpver, plate=plate, ifu=ifu, dir3d=dir3d)
    else:
        img = path.url('mangaimage', drpver=drpver, plate=plate, ifu=ifu, dir3d=dir3d)
    return img
