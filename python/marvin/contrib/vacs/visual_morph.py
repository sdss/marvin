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

import marvin.tools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .base import VACMixIn, VACTarget


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
    version = {'MPL-7': '1.0.1', 'MPL-11': '2.0.1', 'DR16': '1.0.1', 'DR17': '2.0.1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps)

    # Required method
    def set_summary_file(self, release):
        ''' Sets the path to the Visual Morphology summary file '''

        # define the variables to build a unique path to your VAC file
        self.path_params = {"vmver": self.version[release]}

        # get_path returns False if the files do not exist locally
        self.summary_file = self.get_path('mangaVmorpho', path_params=self.path_params)

    # Required method
    def get_target(self, parent_object):
        ''' Accesses VAC data for a specific target from a Marvin Tool object '''

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu

        # download the vac from the SAS if it does not already exist locally
        if not self.file_exists(self.summary_file):
            self.summary_file = self.download_vac('mangaVmorpho', path_params=self.path_params)

        # get path to ancillary VAC files for mosaic images
        self.update_path_params({'plateifu': plateifu})
        desi_mos = self._get_mosaics(self.path_params)

        # create container for more complex return data
        vmdata = VizMorphTarget(plateifu, vacfile=self.summary_file, desi=desi_mos)

        return vmdata

    def _get_mosaics(self, path_params):
        ''' Get a mosaic image file for an image path

        Checks for local existence of the mosaic image filepath.
        If it does not exists, it downloads it.

        Parameters:
            path_params (dict):
                The sdss_access keyword parameters to define a file path

        Returns:
            The mosaic image file path
        '''
        # get the path for the given survey
        mosaic = self.get_path('mangaVmorphoImgs', path_params=path_params)
        # download the mosaic file
        if not self.file_exists(mosaic):
            mosaics = self.download_vac('mangaVmorphoImgs', path_params=path_params)
        # get the path again for the mosaic
        mosaic = self.get_path('mangaVmorphoImgs', path_params=path_params)
        return mosaic


class VizMorphTarget(VACTarget):
    ''' A customized target class to also display morphology mosaics

    This class handles data from both the Visual Morphology summary file and the
    individual image files.  Row data from the summary file for the given target
    is returned via the `data` property.  Images can be displayed via
    the `show_mosaic()` method.

    Parameters:
        targetid (str):
            The plateifu or mangaid designation
        vacfile (str):
            The path of the VAC summary file
        desi (str):
            The path to the image mosaic

    Attributes:
        data:
            The target row data from the main VAC file
        targetid (str):
            The target identifier
    '''

    def __init__(self, targetid, vacfile, desi=None):
        super(VizMorphTarget, self).__init__(targetid, vacfile)
        self._desi_img = desi

    def show_mosaic(self):
        ''' Show the mosaic image for the given survey

        Displays the mosaic image of visual morphology classification
        as a Matplotlib Figure/Axis object.

        Returns:
            A matplotlib axis object
        '''

        impath = self._desi_img
        imdata = mpimg.imread(impath)
        fig, ax = plt.subplots(figsize = (20,5))
        ax.imshow(imdata)
        title = 'Mosaic'
        fig.suptitle(title)
        return ax
