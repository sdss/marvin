#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Brian Cherinka, José Sánchez-Gallego, Brett Andrews
# @Date: Oct 25, 2017
# @Filename: base.py
# @License: BSD 3-Clause
# @Copyright: Brian Cherinka, José Sánchez-Gallego, Brett Andrews


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from astropy import units as u

from marvin.utils.datamodel.maskbit import get_maskbits
from .base import DRPDataModel, DataCube, Spectrum, DRPDataModelList


spaxel_unit = u.Unit('spaxel', represents=u.pixel, doc='A spectral pixel', parse_strict='silent')


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


MPL4 = DRPDataModel('MPL-4', aliases=['MPL4'],
                    datacubes=MPL4_datacubes,
                    spectra=MPL4_spectra,
                    bitmasks=get_maskbits('MPL-4'))

MPL5 = DRPDataModel('MPL-5', aliases=['MPL5'],
                    datacubes=MPL4_datacubes,
                    spectra=MPL4_spectra,
                    bitmasks=get_maskbits('MPL-5'))

MPL6 = DRPDataModel('MPL-6', aliases=['MPL6'],
                    datacubes=MPL4_datacubes + MPL6_datacubes,
                    spectra=MPL4_spectra + MPL6_spectra,
                    bitmasks=get_maskbits('MPL-6'))


datamodel = DRPDataModelList([MPL4, MPL5, MPL6])
