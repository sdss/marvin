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
import marvin.tools
import matplotlib.pyplot as plt
from marvin import log, config
from .base import VACMixIn, VACTarget

class CallableDict(dict):
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if callable(val):
            return val()
        return val

class FIREFLYVAC(VACMixIn):
    """Provides access to the MaNGA-FIREFLY VAC.

    VAC name: FIREFLY

    URL: https://www.sdss.org/dr15/manga/manga-data/manga-firefly-value-added-catalog/

    Description: Returns integrated and resolved stellar population parameters fitted by FIREFLY

    Authors: Justus Neumann, Jianhui Lian, Daniel Thomas, Claudia Maraston, and Lewis Hill

    """

    # Required parameters
    name = 'firefly'
    description = 'Returns stellar population parameters fitted by FIREFLY'
    version = {'DR15': 'v1_1_2', 'DR16': 'v1_1_2','DR17':'v3_1_1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps, marvin.tools.modelcube.ModelCube)

    # Required method
    def set_summary_file(self, release):
        ''' Sets the path to the Firefly summary file '''

        # define the variables to build a unique path to your VAC file
        # look up the MaNGA drpver from the release
        drpver, dapver = config.lookUpVersions(release)
		
        self.path_params = []
        if release=='DR17':
            
            self.path_params.append({'ver': self.version[release], 'drpver': drpver, 'models':'miles'})
            self.path_params.append({'ver': self.version[release], 'drpver': drpver, 'models':'mastar'})
			
            self.summary_file = [self.get_path('mangaffly', path_params=self.path_params[i]) for i in [0,1]]
            
            # specify a special data container for the general VAC tools
            self.data_container = dict(zip(['miles', 'mastar'], self.summary_file))
        else:
            self.path_params.append({'ver': self.version[release], 'drpver': drpver})

            # get_path returns False if the files do not exist locally
            self.summary_file.append(self.get_path('mangaffly', path_params=self.path_params))

    # Required method
    def get_target(self, parent_object):
        ''' Accesses VAC data for a specific target from a Marvin Tool object '''

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu
        imagesz = int(parent_object.header['NAXIS1'])

        if len(self.summary_file)==1:   # for data releases <DR17
            # download the vac from the SAS if it does not already exist locally
            if not self.file_exists(self.summary_file):
                log.info('Warning: This file is ~6 GB.  It may take awhile to download')
                self.summary_file = self.download_vac('mangaffly', path_params=self.path_params)
                
            # create container for more complex return data.
            ffly = FFlyTarget(plateifu, vacfile=self.summary_file,imagesz=imagesz)
            return ffly

        # for DR17 return dictionary; vacfile is only downloaded when key is selected
        ffly = CallableDict({
			"miles":lambda: self.prepare_container(plateifu, imagesz, 0),
			"mastar":lambda: self.prepare_container(plateifu, imagesz, 1)
            })

        return ffly
    
    def prepare_container(self,plateifu,imagesz,n):
        if not self.file_exists(self.summary_file[n]):
            log.info('Warning: This file is ~6 GB.  It may take awhile to download')
            self.summary_file[n] = self.download_vac('mangaffly', path_params=self.path_params[n])
        container = FFlyTarget(plateifu, vacfile=self.summary_file[n], imagesz=imagesz)
        return container


class FFlyTarget(VACTarget):
    ''' A customized target class to also display Firefly 2-d maps

    This class handles data the Firefly summary file.  Row data from the summary
    file for the given target is returned via the `data` property. Specific Firefly
    parameters are available via the `stellar_pops` and `stellar_gradients` methods, respectively.
    2-d maps from the Firefly data can be produced via the `plot_map` method.

    TODO: 

    Parameters:
        targetid (str):
            The plateifu or mangaid designation
        vacfile (str):
            The path of the VAC summary file
        imagesz (int):
            The original array shape of the target cube

    Attributes:
        data:
            The target row data from the main VAC file
        targetid (str):
            The target identifier
    '''

    def __init__(self, targetid, vacfile, imagesz=None):
        super(FFlyTarget, self).__init__(targetid, vacfile)
        self._image_sz = imagesz
        self._parameters = ['lw_age', 'mw_age', 'lw_z', 'mw_z']
        # select the index of the targetid from the main VAC extension
        self._idx = self._get_data(ext='GALAXY_INFO')['plateifu'] == targetid
        

    def stellar_pops(self, parameter=None):
        ''' Returns the global stellar population properties

        Returns the global stellar population property within 1 Re for a given
        stellar population parameter.  If no parameter specified, returns the entire row.

        Parameters:
            parameter (str):
                The stellar population parameter to retrieve.  Can be one of ['lw_age', 'mw_age', 'lw_z', 'mw_z'].
        
        Returns:
            The data from the FIREFLY summary file for the target galaxy
        '''

        if parameter:
            assert parameter in self._parameters, 'parameter must be one of {0}'.format(
                self._parameters
            )

        if not self._indata:
            return "No FIREFLY result exists for {0}".format(self.targetid)

        if parameter:
            return self._get_data(ext='GLOBAL_PARAMETERS')[parameter + '_1re'][self._idx]
        else:
            return self._get_data(ext='GLOBAL_PARAMETERS')[self._idx]

    def stellar_gradients(self, parameter=None):
        ''' Returns the gradient of stellar population properties

        Returns the gradient of the stellar population property for a given
        stellar population parameter.  If no parameter specified, returns the entire row.

        Parameters:
            parameter (str):
                The stellar population parameter to retrieve.  Can be one of ['lw_age', 'mw_age', 'lw_z', 'mw_z'].
        
        Returns:
            The data from the FIREFLY summary file for the target galaxy
        '''

        if parameter:
            assert parameter in self._parameters, 'parameter must be one of {0}'.format(
                self._parameters
            )

        if not self._indata:
            return "No FIREFLY result exists for {0}".format(self.targetid)

        if parameter:
            return self._get_data(ext='GRADIENT_PARAMETERS')[parameter + '_gradient'][self._idx]
        else:
            return self._get_data(ext='GRADIENT_PARAMETERS')[self._idx]

    def _make_map(self, parameter=None):
        ''' Extract and create a 2d map '''

        params = self._parameters + [
            'e(b_v)',
            'stellar_mass',
            'surface_mass_density',
            'signal_noise',
        ]
        assert parameter in params, 'Parameter must be one of {0}'.format(
            params)

        # get the required arrays
        binid = self._get_data(ext='SPATIAL_BINID')[self._idx].squeeze(axis=0)
        bin1d = self._get_data(ext='SPATIAL_INFO')[self._idx, :, 0][0]
        try:
            prop = self._get_data(ext=parameter + '_voronoi')[self._idx, :, 0][0]
        except:
            prop = self._get_data(ext=parameter + '_voronoi')[self._idx, :][0]
        image_sz = self._image_sz

        # make a base map and reshape to a 1d array
        maps = (np.zeros((image_sz, image_sz)) -
                99).reshape(image_sz * image_sz)
        # find relevant elements in bin1d (anything not -9999)
        propinds = np.where(bin1d >= -1)
        # find relevant elements in binid (anything not -9999)
        inds = np.where(binid >= -1)
        # select the relevant elements from binid
        tmp = binid[inds]
        # find non-zero elements from the relevant elements
        newinds = np.where(tmp > -1)[0]
        # map the bin1d indices back to the original binid
        ai = bin1d[propinds].argsort()
        p = ai[np.searchsorted(bin1d[propinds], tmp[newinds], sorter=ai)]
        # replace map elements with the relevant prop parameter
        maps[newinds] = prop[p]
        # reshape the map array from 1d back to 2d
        maps = maps.reshape(image_sz, image_sz)

        return maps

    def plot_map(self, parameter=None, mask=None):
        ''' Plot map of stellar population properties
        
        Plots a 2d map of the specified FIREFLY stellar
        population parameter using Matplotlib.  Optionally mask
        the data when plotting using Numpy's Masked Array.  Default
        is to mask map values < -10.

        Parameters:
            parameter (str):
                The named of the VORONOI stellar pop. parameter
            mask (nd-array):
                A Numpy array of masked values to apply to the map

        Returns:
            The matplotlib axis image object

        '''

        if not self._indata:
            return "No FIREFLY result exists for {0}".format(self.targetid)

        # create the 2d map
        maps = self._make_map(parameter=parameter)

        # only show the spaxels with non-empty values
        mask = (maps < -10) if not mask else mask
        masked_array = np.ma.array(maps, mask=mask)

        # plot the masked map
        fig, ax = plt.subplots()
        axim = ax.imshow(masked_array, interpolation='nearest',
                         cmap='RdYlBu_r', origin='lower')
        ax.set_xlabel('spaxel')
        ax.set_ylabel('spaxel')
        ax.set_title('Firefly {0}'.format(parameter.title()))

        # plot the colour bar
        cbar = fig.colorbar(axim, ax=ax, shrink=0.9)
        cbar.set_label(parameter.title(), fontsize=18, labelpad=20)
        cbar.ax.tick_params(labelsize=22)

        return axim

    def list_parameters(self):
        ''' List the parameters available for plotting '''

        params = self._parameters + [
            'e(b_v)',
            'stellar_mass',
            'surface_mass_density',
            'signal_noise',
        ]
        return params

