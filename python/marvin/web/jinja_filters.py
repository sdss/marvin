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
import numpy as np

'''
This file contains all custom Jinja2 filters for Marvin web
'''


def filtergaltype(value):
    ''' Parse plateifu or mangaid into better form '''
    if value == 'plateifu':
        return 'Plate-IFU'
    elif value == 'mangaid':
        return 'MaNGA-ID'


def isclose(value, newvalue):
    ''' Do a numpy isclose comparison between the two values '''
    return np.isclose(float(value), float(newvalue))


def prettyFlag(value):
    ''' Pretty print bit mask and flags '''
    name, bit, flags = value
    return '{0}: {1} - {2}'.format(name, bit, ', '.join(flags))


def qaclass(value):
    ''' Return an alert indicator based on quality flags '''
    name, bit, flags = value
    isgood = ['VALIDFILE'] == flags
    iscrit = 'CRITICAL' in flags
    out = 'success' if isgood else 'danger' if iscrit else 'warning'
    text = 'Good' if isgood else 'DO NOT USE' if iscrit else 'Warning'
    return out, text


def targtype(value):
    ''' Return the MaNGA target type based on what bit is set '''
    # names = value.get('names', None)
    # namelabel = ', '.join(names)
    # out = namelabel.replace('MNGTRG1', 'Galaxy').replace('MNGTRG2', 'Stellar').replace('MNGTRG3', 'Ancillary')
    out = 'Galaxy' if '1' in value else 'Ancillary' if '3' in value else 'Stellar'
    return out


def split(string, delim=None):
    '''Split a string based on a delimiter'''
    if not delim:
        delim = ' '
    return string.split(delim) if string else None

