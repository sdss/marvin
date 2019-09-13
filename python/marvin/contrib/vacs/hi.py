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

from .base import VACMixIn


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

    #
    # add_plots = ['plot_test']

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


        self.update_path_params({'plateifu': plateifu})
        specfile = self.get_path('mangahispectra', path_params=self.path_params)

        # download the vac from the SAS if it does not already exist locally
        if not self.summary_file:
            self.summary_file = self.download_vac('mangahisum', path_params=self.path_params)

        # create container for more complex return data
        hidata = HIData(plateifu, vacfile=self.summary_file, specfile=specfile)

        # get the spectral data for that row if it exists
        if hidata._indata and not specfile:
            hidata._specfile = self.download_vac('mangahispectra', path_params=self.path_params)

        return hidata


class HIData(object):
    ''' A customized class to handle more complex data

    This class handles data from both the HI summary file and the
    individual HI spectral files.  Row data from the summary file
    is returned via the `data` property.  Spectral data can be plotted via
    the `plot_spectrum` method.

    '''

    def __init__(self, plateifu, vacfile=None, specfile=None):
        self._vacfile = vacfile
        self._specfile = specfile
        self._plateifu = plateifu
        self._hi_data = self._open_file(vacfile)
        self._indata = plateifu in self._hi_data['plateifu']
        self._specdata = None

    def __repr__(self):
        return 'HI({0})'.format(self._plateifu)

    @staticmethod
    def _open_file(hifile):
        return astropy.io.fits.getdata(hifile, 1)

    @property
    def data(self):
        ''' Returns the FITS row data from the mangaHIall summary file '''

        if not self._indata:
            return "No HI data exists for {0}".format(self._plateifu)

        idx = self._hi_data['plateifu'] == self._plateifu
        return self._hi_data[idx]

    def plot_spectrum(self):
        ''' Plot the HI spectrum '''

        if self._specfile:
            if not self._specdata:
                self._specdata = self._open_file(self._specfile)

            vel = self._specdata['VHI'][0]
            flux = self._specdata['FHI'][0]
            spec = Spectrum(flux, unit=u.Jy, wavelength=vel, wavelength_unit=u.km / u.s)
            ax = spec.plot(
                ylabel='HI\ Flux', xlabel='Velocity', title=self._plateifu, ytrim='minmax'
            )
            return ax
        return None

    def plot_massfraction(self):
        ''' Plot the HI mass fraction '''

        drpall = get_drpall_table()
        drpall.add_index('plateifu')
        subset = drpall.loc[self._hi_data['plateifu']]
        log_stmass = np.log10(subset['nsa_elpetro_mass'])
        diff = self._hi_data['logMHI'] - log_stmass
        fig, axes = scatplot(
            log_stmass,
            diff,
            with_hist=False,
            ylim=[-5, 5],
            xlabel=r'log $M_*$',
            ylabel=r'log $M_{HI}/M_*$',
        )
        return axes[0]


# def plot_test(self):
#     print(self.data)
#     drpall = get_drpall_table()
#     drpall.add_index('plateifu')
#     data = self.data[1].data
#     subset = drpall.loc[data['plateifu']]
#     log_stmass = np.log10(subset['nsa_elpetro_mass'])
#     diff = data['logMHI'] - log_stmass
#     fig, axes = scatplot(
#         log_stmass,
#         diff,
#         with_hist=False,
#         ylim=[-5, 5],
#         xlabel=r'log $M_*$',
#         ylabel=r'log $M_{HI}/M_*$',
#     )
#     return axes[0]
