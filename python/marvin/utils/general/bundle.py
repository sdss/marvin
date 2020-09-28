# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: José Sánchez-Gallego
# @Date:   2014-04-18 00:21:02
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-08-08 14:05:43

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import numpy as np
import requests
import warnings
import PIL

from astropy import table
from astropy.io import ascii
from astropy.wcs import WCS
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils import yanny

if sys.version_info.major == 2:
    from cStringIO import StringIO as stringio
else:
    from io import StringIO as stringio
    from io import BytesIO as bytesio


# cached yanny metrology and plateholes objects
SIMOBJ = None
PLATEHOLES = None


class Bundle(object):
    """The location, size, and shape of a MaNGA IFU bundle.

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
        plateifu (str):
            The plateifu of the target
        use_envelope (bool):
            Expands the hexagon area to include an envelope surrounding the hex border
        local (bool):
            If True, grabs the fiducial metrology file from a local MANGACORE

    Attributes:
        fibers (ndarray):
            An Nx3 array of IFU fiber coordinates
        skies (ndarray):
            An Nx2 array of sky fiber coordinates.  Default is unloaded.  Use :meth:`get_sky_coordinates` to load.
        hexagon (ndarray):
            An Nx2 array of coordinates defining the hexagon vertices
    """

    def __init__(self, ra, dec, size=127, use_envelope=True,
                 local=None, plateifu=None, **kwargs):
        self.ra = float(ra)
        self.dec = float(dec)
        self.size = size
        self.plateifu = plateifu
        self.simbMapFile = None
        self.plateholes_file = None
        if self.size not in [7, 19, 37, 61, 91, 127]:
            self.hexagon = None
            return

        self.simbMap = self._get_simbmap_file(local=local)

        # get the fiber coordinates
        self.skies = None
        self.fibers = self.get_fiber_coordinates(size=size)

        # get the hexagon coordinates
        self.hexagon = self._calculate_hexagon(use_envelope=use_envelope)

    def __repr__(self):
        return '<Bundle (ra={0}, dec={1}, ifu={2})>'.format(self.ra, self.dec, self.size)

    def _get_a_file(self, filename, local=None):
        ''' Retrieve a file served locally or remotely over the internet

        Will retrieve a local filename or grab the file
        contents remotely from the SAS

        Parameters:
            filename (str):
                Full filepath to load
            local (bool):
                If True, does a local system check

        Returns:
            A file object to be read in by Yanny
        '''

        if local:
            if not os.path.exists(filename):
                raise MarvinError('No {0} file found locally.'.format(filename))
            else:
                fileobj = filename
        else:
            r = requests.get(filename)
            if not r.ok:
                raise MarvinError('Could not retrieve {0} file remotely'.format(filename))
            else:
                fileobj = stringio(r.content.decode())

        return fileobj

    def _read_in_yanny(self, filename, local=None):
        ''' Read in a yanny file object

        Reads in a local or remote data object as a Yanny file

        Parameters:
            filename (str):
                Full filepath to load
            local (bool):
                If True, does a local system check

        Returns:
            A Yanny object
        '''

        fileobj = self._get_a_file(filename, local=local)

        try:
            data = yanny.yanny(fileobj, np=True)
        except Exception as e:
            raise MarvinError('Cannot read file {0}. {1}'.format(filename, e))

        return data

    def _get_simbmap_file(self, local=None):
        ''' Retrieves the metrology file locally or remotely

        Reads in fresh metrology data or grabs it from
        cache if available

        Parameters:
            local (bool):
                If True, does a local system check

        Returns:
            The metrology object data
        '''

        # create the file path
        rel_path = u'metrology/fiducial/manga_simbmap_127.par'
        if local:
            base_path = os.environ['MANGACORE_DIR']
        else:
            base_path = u'https://svn.sdss.org/public/repo/manga/mangacore/tags/v1_2_3'

        self.simbMapFile = os.path.join(base_path, rel_path)

        # read in the Yanny object
        global SIMOBJ
        if SIMOBJ is None:
            simbmap_data = self._read_in_yanny(self.simbMapFile, local=local)
            SIMOBJ = simbmap_data['SIMBMAP']

        return SIMOBJ

    def _get_plateholes_file(self, plate=None, local=None):
        ''' Retrieves the platesholes file locally or remotely

        Reads in fresh plateHoles data or grabs it from
        cache if available

        Parameters:
            plate (int):
                The plateid to load the plateHoles file for
            local (bool):
                If True, does a local system check

        Returns:
            The plateHoles object data
        '''

        # check for plate
        if not plate and not self.plateifu:
            raise MarvinError('Must have the plate id to get the correct plateholes file')
        elif not plate and self.plateifu:
            plate, ifu = self.plateifu.split('-')

        # create the file path
        pltgrp = '{:04d}XX'.format(int(plate) // 100)
        rel_path = u'platedesign/plateholes/{0}/plateHolesSorted-{1:06d}.par'.format(pltgrp, int(plate))
        if local:
            base_path = os.environ['MANGACORE_DIR']
        else:
            base_path = u'https://svn.sdss.org/public/repo/manga/mangacore/tags/v1_2_3'

        self.platesholes_file = os.path.join(base_path, rel_path)

        # read in the Yanny object
        global PLATEHOLES
        if PLATEHOLES is None:
            plateholes_data = self._read_in_yanny(self.platesholes_file, local=local)
            PLATEHOLES = plateholes_data['STRUCT1']

        return PLATEHOLES

    @staticmethod
    def _get_sky_fibers(data, ifu):
        """ Return the sky fibers associated with each galaxy.

        Parameters:
            data (object):
                the plateHoles object data
            ifu (int):
                The ifu design of the target

        Returns:
            A numpy array of tuples of sky fiber RA, Dec coordinates
        """
        sky_fibers = data[data['targettype'] == 'SKY']
        sky_block = sky_fibers['block'] == int(ifu)
        zip_data = list(zip(sky_fibers[sky_block]['target_ra'], sky_fibers[sky_block]['target_dec']))
        skies = np.array(zip_data)
        return skies

    def get_sky_coordinates(self, plateifu=None):
        ''' Returns the RA, Dec coordinates of the sky fibers

        An Nx2 numpy array, with each row the [RA, Dec] coordinate of
        the sky fiber.

        Parameters:
            plateifu (str):
                The plateifu of the target
        '''

        plateifu = plateifu or self.plateifu
        if not plateifu:
            raise MarvinError('Need the plateifu to get the corresponding sky fibers')
        else:
            plate, ifu = plateifu.split('-')

        data = self._get_plateholes_file(plate=plate)
        self.skies = self._get_sky_fibers(data, ifu)

    def get_fiber_coordinates(self, size=None):
        """ Returns the RA, Dec coordinates for each fiber.

        Parameters:
            size (int):
                The IFU size.  This extracts only the fibers for the given size

        Returns:
            An Nx3 numpy array, each row containing [fiberid, RA, Dec]
        """

        fibreWorld = np.zeros((len(self.simbMap), 3), float)

        raOffDeg = self.simbMap['raoff'] / 3600. / np.cos(self.dec *
                                                          np.pi / 180.)
        decOffDeg = self.simbMap['decoff'] / 3600.

        fibreWorld[:, 0] = self.simbMap['fnumdesign']
        fibreWorld[:, 1] = self.ra + raOffDeg
        fibreWorld[:, 2] = self.dec + decOffDeg

        # extract only the fibers for the specified IFU size
        if size:
            fibreWorld = fibreWorld[fibreWorld[:, 0] <= size]

        return fibreWorld

    def _calculate_hexagon(self, use_envelope=True):
        """ Calculates the vertices of the bundle hexagon. """

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


class Cutout(object):
    """ A Generic SDSS Cutout Image

    Tool which allows to generate an image using the SDSS Skyserver
    Image Cutout service.  See http://skyserver.sdss.org/public/en/help/docs/api.aspx#imgcutout
    for details.

    Parameters:
        ra (float):
            The central Right Ascension of the cutout
        dec (float):
            The central Declination of the cutout
        width (int):
           width of cutout in arcsec
        height (int):
            height in cutout in arcsec
        scale (float):
            pixel scale in arcsec/pixel.  Default is 0.198 "/pix.
        kwargs:
            Allowed boolean keyword arguments to the SDSS Image Cutout Service. Set desired
            keyword parameters to True to enable.

            Available kwargs are:
             - 'photo': PhotoObjs
             - 'bound': BoundingBox
             - 'masks': Masks
             - 'grid': Grid
             - 'outline': Outline
             - 'target': TargetObjs
             - 'fields': Fields
             - 'invert': Invert Image
             - 'spectra': SpecObjs
             - 'label': Label
             - 'plates': Plates

    Attributes:
        rawurl (str):
            The raw url of the cutout
        wcs (WCS):
            The WCS of the generated image
        image (PIL.Image):
            The cutout image object
        size (int):
            The image size in arcsec
        size_pix (int):
            The image size in pixels
        center (tuple):
            The image center RA, Dec

    """

    def __init__(self, ra, dec, width, height, scale=None, **kwargs):
        self.rawurl = ("http://skyserver.sdss.org/public/SkyServerWS/ImgCutout/getjpeg?"
                       "ra={ra}&dec={dec}&scale={scale}&width={width_pix}&height={height_pix}&opt={opts}&query=")
        self.ra = ra
        self.dec = dec
        self.scale = scale or 0.198  # default arcsec/pixel
        self.image = None
        self.center = np.array([ra, dec])
        self.size = np.array([width, height], dtype=int)
        self.coords = {'ra': ra, 'dec': dec,
                       'width': width, 'height': height,
                       'scale': self.scale}
        self._get_pix_size()
        if max(self.size_pix) >= 2048:
            raise MarvinError('Requested image size is too large. '
                              'The Skyserver image cutout can only return a size up to 2048 pixels')

        self._define_wcs()
        self._get_cutout(**kwargs)

    def __repr__(self):
        return ('<Cutout (ra={0}, dec={1}, scale={2}, height={3}, '
                'width={4})>'.format(self.ra, self.dec, self.scale, *self.size_pix))

    def _get_pix_size(self):
        """height,width converted from arcsec->pixels"""
        self.coords['height_pix'] = int(round(self.coords['height'] / self.scale))
        self.coords['width_pix'] = int(round(self.coords['width'] / self.scale))
        self.size_pix = np.array((self.coords['height_pix'], self.coords['width_pix']))

    def _define_wcs(self):
        """
        Given what we know about the scale of the image,
        define a nearly-correct world coordinate system to use with it.
        """
        w = WCS(naxis=2)
        w.wcs.crpix = self.size_pix / 2
        w.wcs.crval = self.center
        w.wcs.cd = np.array([[-1, 0], [0, 1]]) * self.scale / 3600.
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.cunit = ['deg', 'deg']
        w.wcs.radesys = 'ICRS'
        w.wcs.equinox = 2000.0
        self.wcs = w

    def _wcs_to_dict(self):
        ''' Convert and return the WCS as a dictionary'''
        wcshdr = None
        if self.wcs:
            wcshdr = self.wcs.to_header()
            wcshdr = dict(wcshdr)
            wcshdr = {key: str(val) for key, val in wcshdr.items()}
        return wcshdr

    def _make_metadata(self, filetype=None):
        ''' Make the meta data for the image '''

        if 'png' in filetype:
            meta = PIL.PngImagePlugin.PngInfo()
        else:
            meta = None
            warnings.warn('Can only save WCS metadata with PNG filetype', MarvinUserWarning)

        if meta:
            info = {key: str(val) for key, val in self.image.info.items()}
            for row in info:
                meta.add_text(row, info[row])

        return meta

    def _update_info(self):
        ''' Update the image info dictionary '''

        for key, value in self.image.info.items():
            if isinstance(value, tuple):
                self.image.info[key] = value[0]

        wcsdict = self._wcs_to_dict()
        self.image.info = wcsdict
        self.image.info.update(self.coords)
        self.image.info['wdthpix'] = self.image.info.pop('width_pix')
        self.image.info['hghtpix'] = self.image.info.pop('height_pix')

    def _add_options(self, **kwargs):

        allowed = {'grid': 'G', 'label': 'L', 'photo': 'P', 'spectra': 'S',
                   'target': 'T', 'outline': 'O', 'bound': 'B', 'fields': 'F',
                   'masks': 'M', 'plates': 'Q', 'invert': 'I'}

        opts = []
        for key, value in kwargs.items():
            assert key in allowed.keys(), 'Cutout keyword must be one of: {0}'.format(allowed.keys())
            assert isinstance(value, (bool, type(None))), 'Cutout value can only be a Boolean'
            if value:
                opts.append(allowed[key])

        self.coords['opts'] = ''.join(opts)

    def _get_cutout(self, **kwargs):
        """ Gets an image cutout

        Get a cutout around a point, centered at some RA, Dec (in decimal
        degrees), and spanning width,height (in arcseconds) in size.

        Parameters:
            kwargs:
                Allowed keywords into the SDSS Skyserver Image Cutout

        """
        # add options
        self._add_options(**kwargs)
        # retrieve the image
        url = self.rawurl.format(**self.coords)
        response = requests.get(url)
        if not response.ok:
            raise MarvinError('Cannot retrieve image cutout')
        else:
            base_image = response.content
            ioop = stringio if sys.version_info.major == 2 else bytesio
            self.image = PIL.Image.open(ioop(base_image))
            self._update_info()

    def save(self, filename, filetype='png'):
        ''' Save the image cutout to a file

        If the filetype is PNG, it will also save the WCS and coordinate
        information as metadata in the image.

        Parameters:
            filename (str):
                The output filename
            filetype (str):
                The output file extension
        '''

        filename, fileext = os.path.splitext(filename)
        extlist = ['.png', '.bmp', '.jpg', '.jpeg', '.tiff', '.gif', '.ppm']
        assert fileext.lower() in extlist, 'Specified filename not of allowed image type: png, gif, tiff, jpeg, bmp'

        meta = self._make_metadata(filetype=filetype)
        self.image.save(filename, filetype, pnginfo=meta)

    def show(self):
        ''' Show the image cutout '''
        if self.image:
            self.image.show()



