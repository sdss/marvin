#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Brian Cherinka, José Sánchez-Gallego, Brett Andrews
# @Date: Oct 25, 2017
# @Filename: base.py
# @License: BSD 3-Clause
# @Copyright: Brian Cherinka, José Sánchez-Gallego, Brett Andrews


from __future__ import absolute_import, division, print_function

from astropy import units as u

from marvin.utils.datamodel.maskbit import get_maskbits

from .base import RSS, DataCube, DRPCubeDataModel, DRPCubeDataModelList, Spectrum


spaxel_unit = u.Unit('spaxel', represents=u.pixel, doc='A spectral pixel', parse_strict='silent')
fiber_unit = u.Unit('fiber', represents=u.pixel, doc='Spectroscopic fibre', parse_strict='silent')


MPL4_datacubes = [
    DataCube('flux', 'FLUX', 'WAVE', extension_ivar='IVAR',
             extension_mask='MASK', unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
             scale=1e-17, formats={'string': 'Flux'},
             description='3D rectified cube')
]

MPL4_spectra = [
    Spectrum('spectral_resolution', 'SPECRES', extension_wave='WAVE', extension_std='SPECRESD',
             unit=u.Angstrom, scale=1, formats={'string': 'Median spectral resolution'},
             description='Median spectral resolution as a function of wavelength '
                         'for the fibers in this IFU'),
]

MPL6_datacubes = [
    DataCube('dispersion', 'DISP', 'WAVE', extension_ivar=None,
             extension_mask='MASK', unit=u.Angstrom,
             scale=1, formats={'string': 'Dispersion'},
             description='Broadened dispersion solution (1sigma LSF)'),
    DataCube('dispersion_prepixel', 'PREDISP', 'WAVE', extension_ivar=None,
             extension_mask='MASK', unit=u.Angstrom,
             scale=1, formats={'string': 'Dispersion pre-pixel'},
             description='Broadened pre-pixel dispersion solution (1sigma LSF)')

]

MPL6_spectra = [
    Spectrum('spectral_resolution_prepixel', 'PRESPECRES', extension_wave='WAVE',
             extension_std='PRESPECRESD', unit=u.Angstrom, scale=1,
             formats={'string': 'Median spectral resolution pre-pixel'},
             description='Median pre-pixel spectral resolution as a function of '
                         'wavelength for the fibers in this IFU'),
]

RSS_extensions = [
    RSS('xpos', 'XPOS', extension_wave='WAVE', unit=u.arcsec,
        formats={'string': 'Fiber X-positions from the IFU center'},
        description='Array of fiber X-positions relative to the IFU center'),
    RSS('ypos', 'YPOS', extension_wave='WAVE', unit=u.arcsec,
        formats={'string': 'Fiber Y-positions from the IFU center'},
        description='Array of fiber Y-positions relative to the IFU center'),
]


MPL4 = DRPCubeDataModel('MPL-4', aliases=['MPL4', 'v1_5_1'],
                        datacubes=MPL4_datacubes,
                        spectra=MPL4_spectra,
                        bitmasks=get_maskbits('MPL-4'),
                        qual_flag='DRP3QUAL')

MPL5 = DRPCubeDataModel('MPL-5', aliases=['MPL5', 'v2_0_1'],
                        datacubes=MPL4_datacubes,
                        spectra=MPL4_spectra,
                        bitmasks=get_maskbits('MPL-5'),
                        qual_flag='DRP3QUAL')

MPL6 = DRPCubeDataModel('MPL-6', aliases=['MPL6', 'v2_3_1'],
                        datacubes=MPL4_datacubes + MPL6_datacubes,
                        spectra=MPL4_spectra + MPL6_spectra,
                        bitmasks=get_maskbits('MPL-6'),
                        qual_flag='DRP3QUAL')

MPL7 = DRPCubeDataModel('MPL-7', aliases=['MPL7', 'v2_4_3', 'DR15'],
                        datacubes=MPL4_datacubes + MPL6_datacubes,
                        spectra=MPL4_spectra + MPL6_spectra,
                        bitmasks=get_maskbits('MPL-7'),
                        qual_flag='DRP3QUAL')

DR15 = DRPCubeDataModel('DR15', aliases=['DR15', 'v2_4_3'],
                        datacubes=MPL4_datacubes + MPL6_datacubes,
                        spectra=MPL4_spectra + MPL6_spectra,
                        bitmasks=get_maskbits('MPL-7'),
                        qual_flag='DRP3QUAL')

MPL8 = DRPCubeDataModel('MPL-8', aliases=['MPL8', 'v2_5_3'],
                        datacubes=MPL4_datacubes + MPL6_datacubes,
                        spectra=MPL4_spectra + MPL6_spectra,
                        bitmasks=get_maskbits('MPL-8'),
                        qual_flag='DRP3QUAL')

# The DRP Cube Datamodel
datamodel = DRPCubeDataModelList([MPL4, MPL5, MPL6, MPL7, DR15, MPL8])

# Define the RSS Datamodel. Start by copying the Cube datamodel for convenience.
datamodel_rss = datamodel.copy()

for release in datamodel_rss:
    datamodel_rss[release] = datamodel_rss[release].to_rss()

    flux = datamodel_rss[release].rss.flux
    flux.description = 'Row-stacked spectra from all exposures for the target'
    flux.unit = flux.unit * spaxel_unit / fiber_unit
