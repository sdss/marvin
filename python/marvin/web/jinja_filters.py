#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-02-18 18:04:14
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-02-18 18:04:14 by Brian Cherinka
    Last Modified On: 2016-02-18 18:04:14 by Brian

'''
from __future__ import print_function
from __future__ import division

'''
This file contains all custom Jinja2 filters for Marvin web
'''


def filtergaltype(value):
    ''' Parse plateifu or mangaid into better form '''
    if value == 'plateifu':
        return 'Plate-IFU'
    elif value == 'mangaid':
        return 'MaNGA-ID'

