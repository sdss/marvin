#!/usr/bin/env python
# encoding: utf-8
"""
Identify the locations and sizes of MaNGA IFU bundles and individual fibers.

Created by José Sánchez-Gallego on 18 Apr 2014.
Licensed under a 3-clause BSD license.

Revision history:
    18 Apr 2014 J. Sánchez-Gallego
      Initial version

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import requests
from astropy import table
from astropy.io import ascii
from marvin.core.exceptions import MarvinError
from marvin.extern import yanny

if sys.version_info.major == 2:
    from cStringIO import StringIO as stringio
else:
    from io import StringIO as stringio


# cached yanny metrology object
SIMOBJ = None


class Bundle(object):
    """The location, size, and shape of a MaNGA IFU bundle."""

    def __init__(self, ra, dec, size=127, simbmap=None, use_envelope=True,
                 local=None, **kwargs):
        """
        A bundle of a given size at the RA/Dec coordinates.

        Setting envelope creates the hexagon at the outer boundry of the fibers,
        otherwise the hexagon runs through the centers of the outer fibers.

        Parameters:
            ra (float):
                The Right Ascension of the target
            dec (float):
                The Declination of the target
            size (int):
                The fiber size of the IFU, e.g. 19 or 127
            simbmap (str):
                The full filepath to the IFU metrology file
            use_envelope (bool):
                Expands the hexagon area to include an envelope surrounding the hex border
            local (bool):
                If True, grabs the fiducial metrology file from a local MANGACORE
        """

        self.ra = float(ra)
        self.dec = float(dec)
        self.size = size
        self.simbMapFile = None
        if self.size not in [7, 19, 37, 61, 91, 127]:
            self.hexagon = None
            return

        self.simbMap = self._get_the_simbmap_file(simbmap, local=local)

        self.fibers = self.get_fiber_coordinates()
        self.fibers = self.fibers[self.fibers[:, 0] <= size]

        self._calculate_hexagon(use_envelope=use_envelope)

    def __repr__(self):
        return '<Bundle (ra={0}, dec={1}, ifu={2})>'.format(self.ra, self.dec, self.size)

    def _get_the_simbmap_file(self, simbmap, local=None):
        ''' Retrieves the metrology file locally or remotely '''

        # create the file path
        if simbmap is None or simbmap == '':
            rel_path = u'metrology/fiducial/manga_simbmap_127.par'
            if local:
                base_path = os.environ['MANGACORE_DIR']
            else:
                base_path = u'https://svn.sdss.org/public/repo/manga/mangacore/tags/v1_2_3'

            self.simbMapFile = os.path.join(base_path, rel_path)

        # retrieve the file object
        if local:
            if not os.path.exists(self.simbMapFile):
                raise MarvinError('No simbmap metrology file found locally.')
            else:
                simobj = self.simbMapFile
        else:
            r = requests.get(self.simbMapFile)
            if not r.ok:
                raise MarvinError('Could not retrieve metrology file remotely')
            else:
                simobj = stringio(r.content.decode())

        # read in the Yanny object
        global SIMOBJ
        if SIMOBJ is None:
            try:
                simbMap = yanny.yanny(simobj, np=True)['SIMBMAP']
            except Exception as e:
                raise MarvinError('Cannot read simbmap. {0}'.format(e))
            else:
                SIMOBJ = simbMap

        return SIMOBJ

    def get_fiber_coordinates(self):
        """Returns the RA, Dec coordinates for each fibre."""

        fibreWorld = np.zeros((len(self.simbMap), 3), float)

        raOffDeg = self.simbMap['raoff'] / 3600. / np.cos(self.dec *
                                                          np.pi / 180.)
        decOffDeg = self.simbMap['decoff'] / 3600.

        fibreWorld[:, 0] = self.simbMap['fnumdesign']
        fibreWorld[:, 1] = self.ra + raOffDeg
        fibreWorld[:, 2] = self.dec + decOffDeg

        return fibreWorld

    def _calculate_hexagon(self, use_envelope=True):
        """Calculates the vertices of the bundle hexagon."""

        simbMapSize = self.simbMap[self.simbMap['fnumdesign'] <= self.size]

        middleRow = simbMapSize[simbMapSize['decoff'] == 0]
        topRow = simbMapSize[simbMapSize['decoff'] ==
                             np.max(simbMapSize['decoff'])]
        bottopRow = simbMapSize[simbMapSize['decoff'] ==
                                np.min(simbMapSize['decoff'])]

        vertice0 = middleRow[middleRow['raoff'] == np.max(middleRow['raoff'])]
        vertice3 = middleRow[middleRow['raoff'] == np.min(middleRow['raoff'])]

        vertice1 = topRow[topRow['raoff'] == np.max(topRow['raoff'])]
        vertice2 = topRow[topRow['raoff'] == np.min(topRow['raoff'])]

        vertice5 = bottopRow[bottopRow['raoff'] == np.max(bottopRow['raoff'])]
        vertice4 = bottopRow[bottopRow['raoff'] == np.min(bottopRow['raoff'])]

        hexagonOff = np.array(
            [[vertice0['raoff'][0], vertice0['decoff'][0]],
             [vertice1['raoff'][0], vertice1['decoff'][0]],
             [vertice2['raoff'][0], vertice2['decoff'][0]],
             [vertice3['raoff'][0], vertice3['decoff'][0]],
             [vertice4['raoff'][0], vertice4['decoff'][0]],
             [vertice5['raoff'][0], vertice5['decoff'][0]]])

        # This array increases the area of the hexagon so that it is an
        # envelope of the bundle.
        if use_envelope:
            halfFibre = 1.
            hexagonExtra = np.array(
                [[halfFibre, 0.0],
                 [halfFibre, halfFibre],
                 [-halfFibre, halfFibre],
                 [-halfFibre, 0.0],
                 [-halfFibre, -halfFibre],
                 [halfFibre, -halfFibre]])

            hexagonOff += hexagonExtra

        raOffDeg = hexagonOff[:, 0] / 3600. / np.cos(self.dec * np.pi / 180.)
        decOffDeg = hexagonOff[:, 1] / 3600.

        hexagon = hexagonOff.copy()
        hexagon[:, 0] = self.ra + raOffDeg
        hexagon[:, 1] = self.dec + decOffDeg

        self.hexagon = hexagon

        return hexagon

    def create_DS9_regions(self, outputFile=None):
        ''' Writes out the bundle positions into a DS9 region file

        Parameters:
            outputFile (str):
                The output filename
        '''

        if outputFile is None:
            outputFile = 'bundlePositionsDS9.reg'

        radius = 1. / 3600.

        template = """
global color=green font="helvetica 10 normal roman" wcs=wcs
        """
        template.strip().replace('\n', ' ')

        for fibre in self.fibers:
            template += ('\nfk5;circle({0:.10f},{1:.10f},'
                         '{2:.10f}) #text = {{{3}}}').format(
                fibre[1], fibre[2], radius, int(fibre[0]))

        template += '\n'

        out = open(outputFile, 'w')
        out.write(template)
        out.close()

    def print_bundle(self):
        ''' Print the bundle to an Astropy Table '''

        tt = table.Table(self.fibers, names=['fnumdesign', 'RA', 'Dec'],
                         dtype=[int, float, float])
        ascii.write(tt, format='fixed_width_two_line',
                    formats={'RA': '{0:.12f}', 'Dec': '{0:.12f}'})

    def print_hexagon(self):
        ''' Print the hexagon to an Astropy Table '''

        tt = table.Table(self.hexagon, names=['RA', 'Dec'],
                         dtype=[float, float])
        ascii.write(tt, format='fixed_width_two_line',
                    formats={'RA': '{0:.12f}', 'Dec': '{0:.12f}'})
