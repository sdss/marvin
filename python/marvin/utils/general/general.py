from marvin.tools.core.exceptions import MarvinError
from astropy import wcs
import numpy as np

# General utilities

__all__ = ['parseName', 'convertCoords', 'lookUpMpl', 'lookUpVersions']


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

