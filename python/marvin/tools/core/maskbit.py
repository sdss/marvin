#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under a 3-clause BSD license.
#
# map.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-08-22 13:08:00
# @Last modified by:   andrews
# @Last modified time: 2017-08-22 13:08:85


from __future__ import division, print_function,  absolute_import


import numpy as np

import marvin

from marvin.util.dap import datamodel
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils.dap.datamodel.base import Property

class Maskbit(object):
    """Handles maskbits.
    
    Parameters:
        data_product (str):
            Data productDefault is DAP.
        release (str):
            MaNGA data release identifier. Default is ``None``.  If
            ``None``, get release from Marvin config.
    """
    
    def __init__(self, data_product='DAP', release=None):
        
        assert data_product in ['DRP', 'DAP']
        
        if release is None:
            release = config.release

        assert release in allowed_releases
        
        self.data_product = data_product
        self.release = release
        
        self._get_maskbits(data_product=data_product, release=release)
        
    def _get_maskbits(data_product, release):
        """Get maskbits from data product.
        
        Parameters:
            data_product (str):
                Data product to use.
            release (str):
                MaNGA data release identifier.
        """
        
        # get correct product/release
        
        pass

        

