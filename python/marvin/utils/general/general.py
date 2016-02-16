from marvin.tools.core import MarvinError
from astropy import wcs
import numpy as np

# General utilities

__all__ = ['parseName', 'convertCoords']


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


