
import collections
import os
import warnings

import numpy as np
import PIL
from scipy.interpolate import griddata

from astropy import wcs
from astropy import table

import marvin
from marvin import log
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from brain.core.exceptions import BrainError

try:
    from sdss_access import RsyncAccess, AccessError
except ImportError as e:
    RsyncAccess = None

try:
    from sdss_access.path import Path
except ImportError as e:
    Path = None


# General utilities
__all__ = ('convertCoords', 'parseIdentifier', 'mangaid2plateifu', 'findClosestVector',
           'getWCSFromPng', 'convertImgCoords', 'getSpaxelXY',
           'downloadList', 'getSpaxel', 'get_drpall_row', 'getDefaultMapPath',
           'getDapRedux', 'get_nsa_data', '_check_file_parameters')

drpTable = {}


def getSpaxel(cube=True, maps=True, modelcube=True,
              x=None, y=None, ra=None, dec=None, xyorig=None):
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
        coords = np.array([x, y], np.float).T

    elif ra is not None or dec is not None:
        assert x is None and y is None, 'Either use (x, y) or (ra, dec)'
        assert ra is not None and dec is not None, 'Specify both ra and dec'

        inputMode = 'sky'
        isScalar = np.isscalar(ra)
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        coords = np.array([ra, dec], np.float).T

    else:
        raise ValueError('You need to specify either (x, y) or (ra, dec)')

    if not xyorig:
        xyorig = marvin.config.xyorig

    if isinstance(maps, marvin.tools.maps.Maps):
        ww = maps.wcs if inputMode == 'sky' else None
        cube_shape = maps.shape
        plateifu = maps.plateifu
    elif isinstance(cube, marvin.tools.cube.Cube):
        ww = cube.wcs if inputMode == 'sky' else None
        cube_shape = cube.shape
        plateifu = cube.plateifu
    elif isinstance(modelcube, marvin.tools.modelcube.ModelCube):
        ww = modelcube.wcs if inputMode == 'sky' else None
        cube_shape = modelcube.shape
        plateifu = modelcube.plateifu

    iCube, jCube = zip(convertCoords(coords, wcs=ww, shape=cube_shape,
                                     mode=inputMode, xyorig=xyorig).T)

    _spaxels = []
    for ii in range(len(iCube[0])):
        _spaxels.append(
            marvin.tools.spaxel.Spaxel(x=jCube[0][ii], y=iCube[0][ii],
                                       cube=cube, maps=maps, modelcube=modelcube))

    if len(_spaxels) == 1 and isScalar:
        return _spaxels[0]
    else:
        return _spaxels


def convertCoords(coords, mode='sky', wcs=None, xyorig='center', shape=None):
    """Converts input coordinates to array indices.

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
        cubeCoords = np.fliplr(np.array(np.round(cubeCoords[:, :-1]), np.int))

    elif mode in ['pix', 'pixel']:
        assert xyorig, 'if mode==pix, xyorig must be defined.'
        x = coords[:, 0]
        y = coords[:, 1]

        assert shape, 'if mode==pix, shape must be defined.'
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

        cubeCoords = np.array([yCube, xCube], np.int).T

    else:
        raise ValueError('mode must be pix or sky.')

    if shape is not None:
        if ((cubeCoords < 0).any() or
                (cubeCoords[:, 0] > (shape[0] - 1)).any() or
                (cubeCoords[:, 1] > (shape[1] - 1)).any()):
            raise MarvinError('some indices are out of limits.'
                              '``xyorig`` is currently set to "{0}". '
                              'Try setting ``xyorig`` to "{1}".'
                              .format(xyorig, 'center' if xyorig is 'lower' else 'lower'))

    return cubeCoords


def mangaid2plateifu(mangaid, mode='auto', drpall=None, drpver=None):
    """Returns the plate-ifu for a certain mangaid.

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

        if not drpall:
            raise ValueError('no drpall file can be found.')

        # Loads the drpall table if it was not cached from a previos session.
        if drpver not in drpTable:
            drpTable[drpver] = table.Table.read(drpall)

        mangaids = np.array([mm.strip() for mm in drpTable[drpver]['mangaid']])

        plateifus = drpTable[drpver][np.where(mangaids == mangaid)]

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
                marvindb.datadb.PipelineVersion.version == drpver).all()

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
            # response = Interaction('api/general/mangaid2plateifu/{0}/'.format(mangaid))
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
    '''
    Finds the closest vector of array coordinates (x, y) from an input vector of pixel coordinates (x, y).

    Parameters:
        point : tuple
            Original point of interest in pixel units, order of (x,y)
        arr_shape : tuple
            Shape of data array in (x,y) order
        pixel_shape : tuple
            Shape of image in pixels in (x,y) order
        xyorig : str
            Indicates the origin point of coordinates.  Set to "relative" switches to an array coordinate
            system relative to galaxy center.  Default is absolute array coordinates (x=0, y=0) = upper left corner

    Returns:
        minind : tuple
            A tuple of array coordinates in x, y order
    '''

    # set as numpy arrays
    arr_shape = np.array(arr_shape, dtype=int)
    pixel_shape = np.array(pixel_shape, dtype=int)

    # compute midpoints
    xmid, ymid = arr_shape/2
    xpixmid, ypixmid = pixel_shape/2

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
    prod = diff[:, :, 0]*diff[:, :, 1]
    minind = np.unravel_index(prod.argmin(), arr_shape)

    # toggle relative array coordinates
    if xyorig in ['relative', 'center']:
        minind = np.array(minind, dtype=int)
        xmin = minind[0] - xmid
        ymin = ymid - minind[1]
        minind = (xmin, ymin)

    return minind


def getWCSFromPng(image):
    ''' Extracts any WCS info from the metadata of a PNG image

    Extracts the WCS metadata info from the PNG optical
    image of the galaxy using PIL (Python Imaging Library).
    Converts it to an Astropy WCS object.

    Parameters:
        image (str):
            The full path to the image

    Returns:
        pngwcs (WCS):
            an Astropy WCS object

    '''

    pngwcs = None
    try:
        image = PIL.Image.open(image)
    except Exception as e:
        raise MarvinError('Cannot open image {0}: {1}'.format(image, e))

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

    # Close the image
    image.close()

    return pngwcs


def convertImgCoords(coords, image, to_pix=None, to_radec=None):
    ''' Transform the WCS info in an image

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

    '''

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
    ''' Determines if a string input is a plate, plateifu, or manga-id

    Parses a string object id and determines whether it is a
    plate ID, a plate-IFU, or MaNGA-ID designation.

    Parameters:
        galid (str):
            The string of an id name to parse

    Returns:
        idtype (str):
            String indicating either plate, plateifu, mangaid, or None

    '''

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
    """Gets and spaxel from a cube in the DB.

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
        spaxel = mdb.session.query(mdb.datadb.Spaxel).filter_by(cube=cube, x=x, y=y).one()
    except sqlalchemy.orm.exc.NoResultFound as e:
        raise MarvinError('Could not retrieve spaxel for plate-ifu {0} at position {1},{2}: No Results Found: {3}'.format(plateifu, x, y, e))
    except Exception as e:
        raise MarvinError('Could not retrieve cube for plate-ifu {0} at position {1},{2}: Unknown exception: {3}'.format(plateifu, x, y, e))

    return spaxel


def getDapRedux(release=None):
    ''' Retrieve SAS url link to the DAP redux directory

    Parameters:
        release (str):
            The release version of the data to download.
            Defaults to Marvin config.release.

    Returns:
        dapredux (str):
            The full redux path to the DAP MAPS
    '''

    if not Path:
        raise MarvinError('sdss_access is not installed')
    else:
        sdss_path = Path()

    release = release or marvin.config.release
    drpver, dapver = marvin.config.lookUpVersions(release=release)
    # hack a url version of MANGA_SPECTRO_ANALYSIS
    dapdefault = sdss_path.dir('mangadefault', drpver=drpver, dapver=dapver, plate=None, ifu=None)
    dappath = dapdefault.rsplit('/', 2)[0]
    dapredux = sdss_path.url('', full=dappath)
    return dapredux


def getDefaultMapPath(**kwargs):
    ''' Retrieve the default Maps path

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
        bintype (str):
            The bintype of the default file to grab. Defaults to MAPS
        daptype (str):
            The daptype of the default map to grab.  Defaults to SPX-MILESHC

    Returns:
        maplink (str):
            The sas url to download the default maps file
    '''

    if not Path:
        raise MarvinError('sdss_access is not installed')
    else:
        sdss_path = Path()

    # Get kwargs
    release = kwargs.get('release', marvin.config.release)
    drpver, dapver = marvin.config.lookUpVersions(release=release)
    plate = kwargs.get('plate', None)
    ifu = kwargs.get('ifu', None)
    daptype = kwargs.get('daptype', 'SPX-GAU-MILESHC')
    bintype = kwargs.get('bintype', 'MAPS')

    # get the sdss_path name by MPL
    # TODO: this is likely to break in future MPL/DRs. Just a heads up.
    if '4' in release:
        name = 'mangadefault'
    elif '5' in release:
        name = 'mangadap5'
    else:
        return None

    # construct the url link to default maps file
    maplink = sdss_path.url(name, drpver=drpver, dapver=dapver, mpl=release,
                            plate=plate, ifu=ifu, daptype=daptype, mode=bintype)
    return maplink


def downloadList(inputlist, dltype='cube', **kwargs):
    ''' Download a list of MaNGA objects

    Uses sdss_access to download a list of objects
    via rsync.  Places them in your local sas path mimicing
    the Utah SAS.

    i.e. $SAS_BASE_DIR/mangawork/manga/spectro/redux

    Can download cubes, rss files, maps, mastar cubes, png images, default maps, or
    the entire plate directory.

    Parameters:
        inputlist (list):
            Required.  A list of objects to download.  Must be a list of plate IDs,
            plate-IFUs, or manga-ids
        dltype ({'cube', 'map', 'image', 'rss', 'mastar', 'default', 'plate'}):
            Indicated type of object to download.  Can be any of
            plate, cube, imagea, mastar, rss, map, or default (default map).
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
    Returns:
        NA: Downloads

    '''

    # Get some possible keywords
    # Necessary rsync variables:
    #   drpver, plate, ifu, dir3d, [mpl, dapver, bintype, n, mode]
    verbose = kwargs.get('verbose', None)
    as_url = kwargs.get('as_url', None)
    release = kwargs.get('release', marvin.config.release)
    drpver, dapver = marvin.config.lookUpVersions(release=release)
    bintype = kwargs.get('bintype', '*')
    binmode = kwargs.get('binmode', '*')
    daptype = kwargs.get('daptype', '*')
    dir3d = kwargs.get('dir3d', '*')
    n = kwargs.get('n', '*')
    limit = kwargs.get('limit', None)

    # check for sdss_access
    if not RsyncAccess:
        raise MarvinError('sdss_access not installed.')

    # Assert correct dltype
    dltype = 'cube' if not dltype else dltype
    assert dltype in ['plate', 'cube', 'mastar', 'rss', 'map', 'image',
                      'default'], 'dltype must be one of plate, cube, mastar, image, rss, map, default'

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
    elif dltype == 'map':
        if '4' in release:
            name = 'mangamap'
        elif '5' in release:
            name = 'mangadap5'
    elif dltype == 'mastar':
        name = 'mangamastar'
    elif dltype == 'image':
        name = 'mangaimage'

    # create rsync
    rsync_access = RsyncAccess(label='marvin_download', verbose=verbose)
    rsync_access.remote()

    # Add objects
    for item in inputlist:
        if idtype == 'mangaid':
            try:
                plateifu = mangaid2plateifu(item)
            except MarvinError as e:
                plateifu = None
            else:
                plateid, ifu = plateifu.split('-')
        elif idtype == 'plateifu':
            plateid, ifu = item.split('-')
        elif idtype == 'plate':
            plateid = item
            ifu = '*'

        rsync_access.add(name, plate=plateid, drpver=drpver, ifu=ifu, dapver=dapver, dir3d=dir3d,
                         mpl=release, bintype=bintype, n=n, mode=binmode, daptype=daptype)

    # set the stream
    try:
        rsync_access.set_stream()
    except AccessError as e:
        raise MarvinError('Error with sdss_access rsync.set_stream. AccessError: {0}'.format(e))

    # get the list and download
    listofitems = rsync_access.get_urls() if as_url else rsync_access.get_paths()
    rsync_access.commit(limit=limit)


def get_drpall_row(plateifu, drpver=None, drpall=None):
    """Returns a dictionary from drpall matching the plateifu."""

    from marvin import config

    if drpall:
        drpall_table = table.Table.read(drpall)
    else:
        config_drpver, __ = config.lookUpVersions()
        drpver = drpver if drpver else config_drpver
        if drpver not in drpTable:
            drpall = drpall if drpall else config._getDrpAllPath(drpver=drpver)
            drpTable[drpver] = table.Table.read(drpall)
        drpall_table = drpTable[drpver]

    row = drpall_table[drpall_table['plateifu'] == plateifu]

    if len(row) != 1:
        raise ValueError('{0} results found for {1} in drpall table'.format(len(row), plateifu))

    return row[0]


def _db_row_to_dict(row, remove_columns=False):
    """Converts a DB object to a dictionary."""

    from sqlalchemy.inspection import inspect as sa_inspect
    from sqlalchemy.ext.hybrid import hybrid_property
    from sqlalchemy.orm.attributes import InstrumentedAttribute

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
    from marvin.core.core import DotableCaseInsensitive

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
                    sampledb.MangaTarget.mangaid == mangaid).all()

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
                        value = np.asscalar(value)
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

