from marvin.tools.core.exceptions import MarvinError, MarvinUserWarning
from astropy import wcs
import numpy as np
from astropy import table
import warnings

# General utilities

__all__ = ['parseName', 'convertCoords', 'lookUpMpl', 'lookUpVersions',
           'mangaid2plateifu']

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


def convertCoords(x=None, y=None, ra=None, dec=None, shape=None, hdr=None, mode='sky'):
    ''' Convert input coordinates in x,y (relative to galaxy center) or RA, Dec coordinates to
        array indices x, y for spaxel extraction. Returns as xCube, yCube '''

    if mode == 'sky':
        ra = float(ra)
        dec = float(dec)
        cubeWCS = wcs.WCS(hdr)
        xCube, yCube, __ = cubeWCS.wcs_world2pix([[ra, dec, 1.]], 1)[0]
    else:
        x = float(x)
        y = float(y)
        yMid, xMid = np.array(shape) / 2.
        xCube = int(xMid + x)
        yCube = int(yMid - y)

    return xCube, yCube

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
    from marvin import config
    from marvin import datadb, session
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

        if not session or not datadb:
            raise MarvinError('no DB connection found')

        if not drpver:
            raise MarvinError('drpver not set.')

        cubes = session.query(datadb.Cube).join(
            datadb.PipelineInfo, datadb.PipelineVersion).filter(
                datadb.Cube.mangaid == mangaid,
                datadb.PipelineVersion.version == drpver).all()

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
