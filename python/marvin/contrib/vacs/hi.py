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
from marvin.tools.vacs import VACs
from marvin import log

from .base import VACMixIn, VACTarget

def choose_best_spectrum(par1, par2, conf_thresh=0.1):
    '''choose optimal HI spectrum based on the following criteria:
    (1) If both detected and unconfused, choose highest SNR
    (2) If both detected and both confused, choose lower confusion prob.
    (3) If both detected and one confused, choose non-confused
    (4) If one non-confused detection and one non-detection, go with detection
    (5) If one confused detetion and one non-detection, go with non-detection
    (6) If niether detected, choose lowest rms

    par1 and par2 are dictionaries with the following parameters:
    program - gbt or alfalfa
    snr - integrated SNR
    rms - rms noise level
    conf_prob - confusion probability

    conf_thresh = maximum confusion probability below which we classify
    the object as essentially unconfused. Default to 0.1 following
    (Stark+21)

    '''

    programs = [par1['program'],par2['program']]
    sel_high_snr = np.argmax([par1['snr'],par2['snr']])
    sel_low_rms = np.argmin([par1['rms'],par2['rms']])
    sel_low_conf = np.argmin([par1['conf_prob'],par2['conf_prob']])
    

     #both detected
    if (par1['snr'] > 0) & (par2['snr'] > 0):
        if (par1['conf_prob'] <= conf_thresh) & (par2['conf_prob'] <= conf_thresh):
            pick = sel_high_snr
        elif (par1['conf_prob'] <= conf_thresh) & (par2['conf_prob'] > conf_thresh):
            pick = 0
        elif (par1['conf_prob'] > conf_thresh) & (par2['conf_prob'] <= conf_thresh):
             pick = 1
        elif (par1['conf_prob'] > conf_thresh) & (par2['conf_prob'] > conf_thresh):
            pick = sel_low_conf

    #both nondetected
    elif (par1['snr'] <= 0) & (par2['snr'] <= 0):
        pick = sel_low_rms

    #one detected
    elif (par1['snr'] > 0) & (par2['snr'] <= 0):
        if par1['conf_prob'] < conf_thresh:
            pick=0
        else:
            pick=1
    elif (par1['snr'] <= 0) & (par2['snr'] > 0):
        if par2['conf_prob'] < conf_thresh:
            pick=1
        else:
            pick=0

    return programs[pick]
            
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
    #version = {'MPL-7': 'v1_0_1', 'DR15': 'v1_0_1', 'DR16': 'v1_0_2', 'DR17': 'v2_0_1'}
    version = {'MPL-7': 'v1_0_1', 'DR15': 'v1_0_1', 'DR16': 'v1_0_2', 'DR17': 'v2_0_1', 'MPL-11': 'v2_0_1'}


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

    def set_program(self,plateifu):
        print('does this work?')
        # download the vac from the SAS if it does not already exist locally
        if not self.file_exists(self.summary_file):
            self.summary_file = self.download_vac('mangahisum', path_params=self.path_params)

        #find all entries in summary file with this plate-ifu. I need the full summary file here. Find best entry between GBT/ALFALFA based on dept and confusion. Then update self.path_params['program'] with alfalfa or gbt.
        
        #summary = VACs.HI.get_table()
        summary = VACs().HI.data[1].data
        galinfo = summary[summary['plateifu'] == plateifu]
        if len(galinfo) == 1:
            if galinfo['session']=='ALFALFA':
                program = 'alfalfa'
            else:
                program = 'gbt'
        else:
            par1={'program':'gbt','snr':0.,'rms':galinfo[0]['rms'],'conf_prob':galinfo[0]['conf_prob']}
            par2={'program':'gbt','snr':0.,'rms':galinfo[1]['rms'],'conf_prob':galinfo[1]['conf_prob']}
            if galinfo[0]['session']=='ALFALFA':
                par1['program']='alfalfa'
            if galinfo[1]['session']=='ALFALFA':
                par2['program']='alfalfa'
            if galinfo[0]['fhi'] > 0:
                par1['snr'] = galinfo[0]['fhi']/galinfo[0]['efhi']
            if galinfo[1]['fhi'] > 0:
                par2['snr'] = galinfo[1]['fhi']/galinfo[1]['efhi']

            program = choose_best_spectrum(par1,par2)

        log.info('Using data from {0}'.format(program))
        #print('Using data from program: ',program)
            
        # get path to ancillary VAC file for target HI spectra
        self.update_path_params({'program':program})
        
    # Required method
    def get_target(self, parent_object):
        ''' Accesses VAC data for a specific target from a Marvin Tool object '''

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu
        self.update_path_params({'plateifu': plateifu})

        print(parent_object.release)
        
        if parent_object.release in ['DR17', 'MPL-11']:
            print('is dr17')
            self.set_program(plateifu)
            # # download the vac from the SAS if it does not already exist locally
            # if not self.file_exists(self.summary_file):
            #     self.summary_file = self.download_vac('mangahisum', path_params=self.path_params)
    
            # #find all entries in summary file with this plate-ifu. I need the full summary file here. Find best entry between GBT/ALFALFA based on dept and confusion. Then update self.path_params['program'] with alfalfa or gbt.
            
            # #summary = VACs.HI.get_table()
            # summary = VACs().HI.data[1].data
            # galinfo = summary[summary['plateifu'] == plateifu]
            # if len(galinfo) == 1:
            #     if galinfo['session']=='ALFALFA':
            #         program = 'alfalfa'
            #     else:
            #         program = 'gbt'
            # else:
            #     par1={'program':'gbt','snr':0.,'rms':galinfo[0]['rms'],'conf_prob':galinfo[0]['conf_prob']}
            #     par2={'program':'gbt','snr':0.,'rms':galinfo[1]['rms'],'conf_prob':galinfo[1]['conf_prob']}
            #     if galinfo[0]['session']=='ALFALFA':
            #         par1['program']='alfalfa'
            #     if galinfo[1]['session']=='ALFALFA':
            #         par2['program']='alfalfa'
            #     if galinfo[0]['fhi'] > 0:
            #         par1['snr'] = galinfo[0]['fhi']/galinfo[0]['efhi']
            #     if galinfo[1]['fhi'] > 0:
            #         par2['snr'] = galinfo[1]['fhi']/galinfo[1]['efhi']
    
            #     program = choose_program(par1,par2)
    
            # log.info('Using data from '.format(program))
            # #print('Using data from program: ',program)
                
            # # get path to ancillary VAC file for target HI spectra
            # self.update_path_params({'program':program})
        specfile = self.get_path('mangahispectra', path_params=self.path_params)
        print(specfile)
        
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
                ylabel='HI\ Flux\ Density', xlabel='Velocity', title=self.targetid, ytrim='minmax'
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
