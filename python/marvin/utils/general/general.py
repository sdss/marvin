from marvin.tools.core.exceptions import MarvinError, MarvinUserWarning
from astropy import wcs
import numpy as np
from astropy import table
from scipy.interpolate import griddata
import warnings
import marvin
import os
import PIL

# General utilities

__all__ = ['parseName', 'convertCoords', 'lookUpMpl', 'lookUpVersions',
           'mangaid2plateifu', 'findClosestVector', 'getWCSFromPng', 'convertImgCoords',
           'getSpaxelXY', 'getSpaxelAPI']

drpTable = None


def parseName(name):
    ''' Parse a string name of either manga-id or plate-ifu '''
    try:
        namesplit = name.split('-')
    except AttributeError as e:
        raise AttributeError('Could not split on input name {0}: {1}'.format(name, e))

    if len(namesplit) == 1:
        raise MarvinError('Input name not of type manga-id or plate-ifu')
    else:
        if len(namesplit[0]) == 4:
            plateifu = name
            mangaid = None
        else:
            mangaid = name
            plateifu = None

    return mangaid, plateifu


def convertCoords(coords, mode='sky', wcs=None, xyorig='center', shape=None):
    """Converts input coordinates to array indices.

    Converts input positions in x, y or RA, Dec coordinates to array indices
    (in Numpy style) or spaxel extraction. In case of pixel coordinates, the
    origin of reference (either the center of the cube or the lower left
    corner) can be specified via `xyorig`.

    If `shape` is defined (mandatory for `mode='pix'`, optional for
    `mode='sky'`) and one or more of the resulting indices are outside the
    size of the input shape, an error is raised.

    This functions is mostly intended for internal use.

    Parameters
    ----------
    coords : `np.array`
        The input coordinates, as an array of shape Nx2.
    mode : str
        The type of input coordinates, either `'sky'` for celestial coordinates
        (FK5), or `'pix'` for pixel coordinates.
    wcs : None or `astropy.wcs.WCS` object
        If `mode='sky'`, the WCS solution from which the cube coordinates can
        be derived.
    xyorig : str
        If `mode='pix'`, the reference point from which the coordinates are
        measured. Valid values are `'center'`, for the centre of the spatial
        dimensions of the cube, or `'lower'` for the lower-left corner.
    shape : None or `np.array`
        If `mode='pix'`, the shape of the spatial dimensions of the cube,
        so that the central position can be calculated.

    Returns
    -------
    result : `np.array`
        An array with the same shape as `coords`, containing the cube index
        positions for the input coordinates, in Numpy style (i.e., the first
        element being the row and the second the column).

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
            yCube = np.round(yMid - y)
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
            raise MarvinError('some indices are out of limits.')

    return cubeCoords


mpldict = {'MPL-4': ('v1_5_1', '1.1.1'), 'MPL-3': ('v1_3_3', 'v1_0_0'), 'MPL-2': ('v1_2_0', None), 'MPL-1': ('v1_0_0', None)}


def lookUpVersions(mplver):
    ''' Retrieve the DRP and DAP versions that make up an MPL '''

    try:
        drpver, dapver = mpldict[mplver]
    except KeyError as e:
        raise MarvinError('MPL version {0} not found in lookup table. No associated DRP/DAP versions. Should they be added?  Check for typos.'.format(mplver))

    return drpver, dapver


def lookUpMpl(drpver):
    ''' Retrieve the MPL version for a given DRP version'''

    # Flip the mpldict
    verdict = {val[0]: key for key, val in mpldict.items()}

    try:
        mplver = verdict[drpver]
    except KeyError as e:
        raise MarvinError('DRP version {0} not found in lookup table. No associated MPL version. Should one be added?  Check for typos.'.format(drpver))

    return mplver


def mangaid2plateifu(mangaid, mode='auto', drpall=None, drpver=None):
    """Returns the plate-ifu for a certain mangaid.

    Uses either the DB or the drpall file to determine the plate-ifu for
    a mangaid. If more than one plate-ifu are available for a certain ifu,
    and `mode='drpall'`, the one with the higher SN2 (calculated as the sum of
    redSN2 and blueSN2) will be used. If `mode='db'`, the most recent one will
    be used.

    Parameters
    ----------
    mangaid : str
        The mangaid for which the plate-ifu will be returned.
    mode : str
        Either `'auto'`, `'drpall'`, `'db'`, or `'remote'`. If `'drpall'` or
        `'db'`, the  drpall file or the local database, respectively, will be
        used. If `'remote'`, a request to the API will be issued. If `'auto'`,
        the local modes will be tried before the remote mode.
    drpall : str or None
        The path to the drpall file to use. If None, the file in
        `config.drpall` will be used.
    drpver : str or None
        The DRP version to use. If None, the one in `config.drpver` will be
        used. If `drpall` is defined, this value is ignored.

    Returns
    -------
    plateifu : str
        The plate-ifu string for the input `mangaid`.

    """

    global drpTable
    from marvin import config, marvindb
    from marvin.api.api import Interaction

    # The modes and order over which the auto mode will loop.
    autoModes = ['db', 'drpall', 'remote']

    assert mode in autoModes + ['auto'], 'mode={0} is not valid'.format(mode)

    drpver = drpver if drpver else config.drpver
    drpall = drpall if drpall else config._getDrpAllPath(drpver=drpver)

    if mode == 'drpall':

        if not drpall:
            raise ValueError('no drpall file can be found.')

        # Loads the drpall table if it was not cached from a previos session.
        if not drpTable:
            drpTable = table.Table.read(drpall)

        mangaids = np.array([mm.strip() for mm in drpTable['mangaid']])

        plateifus = drpTable[np.where(mangaids == mangaid)]

        if len(plateifus) > 1:
            warnings.warn('more than one plate-ifu found for mangaid={0}. '
                          'Using the one with the highest SN2.'.format(mangaid), MarvinUserWarning)
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
            warnings.warn('more than one plate-ifu found for mangaid={0}. Using a the one with the higest SN2'.format(mangaid), MarvinUserWarning)
            total_sn2 = [float(cube.header['BLUESN2'])+float(cube.header['REDSN2']) for cube in cubes]
            cube = cubes[np.argmax(total_sn2)]
        else:
            cube = cubes[0]

        return '{0}-{1}'.format(cube.plate, cube.ifu.name)

    elif mode == 'remote':

        response = Interaction('api/general/mangaid2plateifu/{0}/'.format(mangaid))
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


def getDbMachine():
    ''' Get the machine that the app is running on.  This determines correct database and app configuration '''
    # Get machine
    machine = os.environ.get('HOSTNAME', None)

    # Check if localhost or not
    try:
        localhost = bool(os.environ['MANGA_LOCALHOST'])
    except:
        localhost = machine == 'manga'

    # Check if Utah or not
    try:
        utah = os.environ['UUFSCELL'] == 'kingspeak.peaks'
    except:
        utah = None
    # Check if sas-vm or not
    sasvm = 'sas-vm' in machine if machine else None

    # Set the dbconfig variable
    if localhost:
        return 'local'
    elif utah or sasvm:
        return 'utah'
    else:
        return None


def convertIvarToErr(ivar):
    ''' Converts a list of inverse variance into an a list of standard errors '''

    assert type(ivar) == list or type(ivar) == np.ndarray, 'Input ivar is not of type list or an Numpy ndarray'

    if type(ivar) == list:
        ivar = np.array(ivar)

    error = np.zeros(ivar.shape)
    notnull = ivar != 0.0
    error[notnull] = 1/np.sqrt(ivar[notnull])
    error = list(error)
    return error


def findClosestVector(point, arr_shape=None, pixel_shape=None, xyorig=None):
    '''
    Finds the closest vector of array coordinates (x, y) from an input vector of pixel coordinates (x, y).

    Parameters:
    ----------
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
    --------
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
    if xyorig == 'relative':
        minind = np.array(minind, dtype=int)
        xmin = minind[0] - xmid
        ymin = ymid - minind[1]
        minind = (xmin, ymin)

    return minind


def getWCSFromPng(image):
    ''' Extracts any WCS info from the metadata of a PNG image '''

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

    return pngwcs


def convertImgCoords(coords, image, to_pix=None, to_radec=None):
    ''' Convert image pixel coordinates to RA/Dec based on PNG image metadata or vice_versa'''

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


def getSpaxelXY(cube, plateifu, x, y):

    import sqlalchemy

    mdb = marvin.marvindb

    try:
        spaxel = mdb.session.query(mdb.datadb.Spaxel).filter_by(
            cube=cube, x=x, y=y).one()
    except sqlalchemy.orm.exc.NoResultFound as e:
        raise MarvinError(
            'Could not retrieve spaxel for plate-ifu {0} at position {1},{2}: '
            'No Results Found: {3}'.format(plateifu, x, y, e))
    except Exception as e:
        raise MarvinError(
            'Could not retrieve cube for plate-ifu {0} at position'
            ' {1},{2}: Unknown exception: {3}'.format(plateifu, x, y, e))

    return spaxel


def getSpaxelAPI(coord1, coord2, mangaid, mode='pix', ext='flux',
                 xyorig='center'):

    from marvin.api.api import Interaction

    # Parse the variables into right frame

    path = '{0}={1}/{2}={3}/ext={4}/xyorig={5}'.format(
        'x' if mode == 'pix' else 'ra', coord1,
        'y' if mode == 'pix' else 'dec', coord2, ext, xyorig)

    routeparams = {'name': mangaid, 'path': path}

    # Get the getSpectrum Route
    url = marvin.config.urlmap['api']['getspectra']['url'].format(
        **routeparams)

    # Make the API call
    response = Interaction(url)

    if response.status_code == 200:
        if response.results['status'] == 1:
            return response.getData()
        else:
            raise MarvinError('Could not retrieve spaxels remotely: {0}'
                              .format(response.results['error']))
    else:
        raise MarvinError(
            'Error retrieving response: Http status code {0}: {1}'
            .format(response.status_code, response.results['message']))
