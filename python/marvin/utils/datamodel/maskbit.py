#!/usr/bin/env python
# encoding: utf-8
#
# maskbit.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-10-06 17:10:00
# @Last modified by:   andrews
# @Last modified time: 2017-11-15 10:11:29

from __future__ import division, print_function, absolute_import

import pandas as pd

from marvin.utils.general.maskbit import Maskbit

__ALL__ = ('get_maskbits')


def get_maskbits(release):
    maskbits = {
        'MANGA_TARGET1': Maskbit('MANGA_TARGET1', description='Targeting bits for all galaxy targets.'),
        'MANGA_TARGET2': Maskbit('MANGA_TARGET2', description='Targeting bits for all non-galaxy targets.'),
        'MANGA_TARGET3': Maskbit('MANGA_TARGET3', description='Targeting bits for ancillary targets.'),
        'MANGA_DRP3QUAL': Maskbit('MANGA_DRP3QUAL', description='Describes the quality of the final cubes and RSS files.'),
        'MANGA_DAPQUAL': Maskbit('MANGA_DAPQUAL', description='Describes the overall quality of the data.'),
        'MANGA_DRP3PIXMASK': Maskbit('MANGA_DRP3PIXMASK', description='Describes whether a given spaxel should be used for science analyses.'),
        'MANGA_DAPPIXMASK': Maskbit('MANGA_DAPPIXMASK',
                                    description='2d image bitmap used to describe the quality of '
                                    'individual pixel measurements in the DAP MAPS file.'),
        'MANGA_DAPSPECMASK': Maskbit('MANGA_DAPSPECMASK',
                                     description='3d cube bitmap used to describe the quality of '
                                     'individual spaxel fits in the DAP model data cube file.')
        }

    if release == 'MPL-4':
        maskbits['MANGA_DAPPIXMASK'] = MPL4_dappixmask()
        __ = maskbits.pop('MANGA_DAPQUAL')

    return maskbits


def MPL4_dappixmask():
    schema = pd.DataFrame([(0, 'DONOTUSE', 'Do not use this spaxel for science')],
                          columns=['bit', 'label', 'description'])
    return Maskbit(name='MANGA_DAPPIXMASK', schema=schema,
                   description='2d image bitmap used to describe the quality of individual pixel '
                               'measurements in the DAP MAPS file.')
