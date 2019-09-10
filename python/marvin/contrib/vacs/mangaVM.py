# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-10-11 17:51:43
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-29 17:23:15

from __future__ import print_function, division, absolute_import

import astropy
import marvin.tools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .base import VACMixIn


class VMORPHOVAC(VACMixIn):
    """Provides access to the MaNGA-VISUAL-MORPHOLOGY VAC.

    VAC name: manga_visual_morpho

    URL: https://data.sdss.org/datamodel/files/MANGA_MORPHOLOGY/visual_morpho/manga_Vmorpho.html

    Description: A new morphology catalogue is presented in this VAC, based on a pure visual
    morphological classification. This catalogue contains the T-Type morphology, visual
    attributes (barred, edge-on, tidal debris) and the CAS parameters (Concentration, Asymmetry
    and Clumpiness; from the DESI images.

    Authors: J. Antonio Vazquez-Mata and Hector Hernandez-Toledo

    """

    # Required parameters
    name = 'visual_morphology'
    description = 'Returns visual morphology data'
    version = {'DR16': '1.0.1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps)

    # Required method
    def set_summary_file(self, release):
        path_params = {"vmver": self.version[release]}
        vmfile = self.get_path('mangaVmorpho', path_params=path_params)
        self.summary_file = vmfile

    # Required method
    def get_data(self, parent_object):

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu
        release = parent_object.release

        # define the variables to build a unique path to your VAC file
        path_params = {'vmver': self.version[release], 'plateifu': plateifu, 'survey': '*'}

        # get_path returns False if the files do not exist locally
        vmfile = self.get_path('mangaVmorpho', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not vmfile:
            vmfile = self.download_vac('mangaVmorpho', path_params=path_params)

        # get the mosaic images
        sdss_mos, desi_mos = self._get_mosaics(path_params)

        # create container for more complex return data
        vmdata = VizMorpho(plateifu, vmfile=vmfile, sdss=sdss_mos, desi=desi_mos)

        return vmdata

    def _get_mosaics(self, path_params):
        ''' Get the mosaic images for SDSS and DESI surveys
        
        Parameters:
            path_params (dict):
                The sdss_access keyword parameters to define a file path
        
        Returns:
            The SDSS and DESI local image filepaths
        '''
        sdss_mosaic = self._check_mosaic('sdss', path_params)
        desi_mosaic = self._check_mosaic('desi', path_params)
        return sdss_mosaic, desi_mosaic

    def _check_mosaic(self, survey, path_params):
        ''' Get a mosaic image file for a survey path
        
        Checks for local existence of the mosaic image filepath.
        If it does not exists, it downloads it. 

        Parameters:
            survey (str):
                The survey to download.  Either sdss or desi
            path_params (dict):
                The sdss_access keyword parameters to define a file path
        
        Returns:
            The mosaic image file path
        '''
        # get the path for the given survey
        path_params['survey'] = survey
        mosaic = self.get_path('mangaVmorphoImgs', path_params=path_params)
        # download the mosaic file (downloads both surveys at once)
        if not mosaic:
            pp = path_params.copy()
            pp['survey'] = '*'
            mosaics = self.download_vac('mangaVmorphoImgs', path_params=pp)
        # get the path again for the single survey
        mosaic = self.get_path('mangaVmorphoImgs', path_params=path_params)
        return mosaic


class VizMorpho(object):
    ''' A customized class to handle more complex data

    This class handles data from both the Visual Morphology summary file and the
    individual image files.  Row data from the summary file
    is returned via the `data` property.  Images can be displayed via
    the `show_mosaic` method.

    '''

    def __init__(self, plateifu, vmfile=None, sdss=None, desi=None):
        self._vmfile = vmfile
        self._plateifu = plateifu
        self._sdss_img = sdss
        self._desi_img = desi
        self._vm_data = self._open_file(vmfile)
        self._indata = plateifu in self._vm_data['plateifu']
        self._specdata = None

    def __repr__(self):
        return 'VisualMorpho({0})'.format(self._plateifu)

    @staticmethod
    def _open_file(vmfile):
        return astropy.io.fits.getdata(vmfile, 1)

    @property
    def data(self):
        ''' Returns the FITS row data from the visual morphology summary file '''

        if not self._indata:
            return "No morphology data exists for {0}".format(self._plateifu)

        idx = self._vm_data['plateifu'] == self._plateifu
        return self._vm_data[idx]

    def show_mosaic(self, survey=None):
        ''' Show the mosaic image for the given survey

        Displays the mosaic image of visual morphology classification
        for the given survey as a Matplotlib Figure/Axis object.

        Parameters:
            survey (str):
                The survey name.  Can be either "sdss" or "desi".
        
        Returns:
            A matplotlib axis object
        '''

        assert survey in ['sdss', 'desi'], 'Must specify either survey: sdss or desi'

        impath = self._sdss_img if survey == 'sdss' else self._desi_img
        imdata = mpimg.imread(impath)
        fig, ax = plt.subplots()
        ax.imshow(imdata)
        title = '{0} Mosaic'.format(survey.upper())
        fig.suptitle(title)
        return ax
