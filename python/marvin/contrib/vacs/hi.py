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

import numpy as np
import astropy
import astropy.units as u
import marvin.tools
from marvin.tools.quantities.spectrum import Spectrum
from marvin.utils.general.general import get_drpall_table
from marvin.utils.plot.scatter import plot as scatplot

from .base import VACMixIn, VACTarget


class HIVAC(VACMixIn):
    """Provides access to the MaNGA-HI VAC.

    VAC name: HI

    URL: link

    Description: Returns HI summary data and spectra

    Authors: David Stark and Karen Masters

    """

    # Required parameters
    name = 'HI'
    description = 'Returns HI summary data and spectra'
    version = {'MPL-7': 'v1_0_1', 'DR15': 'v1_0_1', 'DR16': 'v1_0_2'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps, marvin.tools.modelcube.ModelCube)

    # optional methods to attach to your main VAC tool in ~marvin.tools.vacs.VACs
    add_methods = ['plot_mass_fraction']

    # Required method
    def set_summary_file(self, release):
        ''' Sets the path to the HI summary file '''

        # define the variables to build a unique path to your VAC file
        self.path_params = {'ver': self.version[release], 'type': 'all', 'program': 'GBT16A_095'}

        # get_path returns False if the files do not exist locally
        self.summary_file = self.get_path("mangahisum", path_params=self.path_params)

    # Required method
    def get_target(self, parent_object):
        ''' Accesses VAC data for a specific target from a Marvin Tool object '''

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu

        # download the vac from the SAS if it does not already exist locally
        if not self.file_exists(self.summary_file):
            self.summary_file = self.download_vac('mangahisum', path_params=self.path_params)

        # get path to ancillary VAC file for target HI spectra
        self.update_path_params({'plateifu': plateifu})
        specfile = self.get_path('mangahispectra', path_params=self.path_params)

        # create container for more complex return data
        hidata = HITarget(plateifu, vacfile=self.summary_file, specfile=specfile)

        # get the spectral data for that row if it exists
        if hidata._indata and not self.file_exists(specfile):
            hidata._specfile = self.download_vac('mangahispectra', path_params=self.path_params)

        return hidata


class HITarget(VACTarget):
    ''' A customized target class to also display HI spectra

    This class handles data from both the HI summary file and the
    individual spectral files.  Row data from the summary file for the given target
    is returned via the `data` property.  Spectral data can be displayed via
    the the `plot_spectrum` method.

    Parameters:
        targetid (str):
            The plateifu or mangaid designation
        vacfile (str):
            The path of the VAC summary file
        specfile (str):
            The path to the HI spectra

    Attributes:
        data:
            The target row data from the main VAC file
        targetid (str):
            The target identifier
    '''

    def __init__(self, targetid, vacfile, specfile=None):
        super(HITarget, self).__init__(targetid, vacfile)
        self._specfile = specfile
        self._specdata = None

    def plot_spectrum(self):
        ''' Plot the HI spectrum '''

        if self._specfile:
            if not self._specdata:
                self._specdata = self._get_data(self._specfile)

            vel = self._specdata['VHI'][0]
            flux = self._specdata['FHI'][0]
            spec = Spectrum(flux, unit=u.Jy, wavelength=vel,
                            wavelength_unit=u.km / u.s)
            ax = spec.plot(
                ylabel='HI\ Flux', xlabel='Velocity', title=self.targetid, ytrim='minmax'
            )
            return ax
        return None

#
# Functions to become available on your VAC in marvin.tools.vacs.VACs


def plot_mass_fraction(vacdata_object):
    ''' Plot the HI mass fraction
    
    Computes and plots the HI mass fraction using
    the NSA elliptical Petrosian stellar mass from the
    MaNGA DRPall file.  Only plots data for subset of
    targets in both the HI VAC and the DRPall file.

    Parameters:
        vacdata_object (object):
            The `~.VACDataClass` instance of the HI VAC 

    Example:
        >>> from marvin.tools.vacs import VACs
        >>> v = VACs()
        >>> hi = v.HI
        >>> hi.plot_mass_fraction()
    '''
    drpall = get_drpall_table()
    drpall.add_index('plateifu')
    data = vacdata_object.data[1].data
    subset = drpall.loc[data['plateifu']]
    log_stmass = np.log10(subset['nsa_elpetro_mass'])
    diff = data['logMHI'] - log_stmass
    fig, axes = scatplot(
        log_stmass,
        diff,
        with_hist=False,
        ylim=[-5, 5],
        xlabel=r'log $M_*$',
        ylabel=r'log $M_{HI}/M_*$',
    )
    return axes[0]
