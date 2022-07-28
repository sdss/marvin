#!/usr/bin/env python
# encoding: utf-8

# Created by Brian Cherinka on 2016-02-18 18:04:14
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-02-18 18:04:14 by Brian Cherinka
#     Last Modified On: 2016-02-18 18:04:14 by Brian

from __future__ import print_function
from __future__ import division
import numpy as np
import flask
import jinja2


# If the filter is to return HTML code and you don't want it autmatically
# escaped, return the value as "return Markup(value)".

jinjablue = flask.Blueprint('jinja_filters', __name__)

# Ref: http://stackoverflow.com/questions/12288454/how-to-import-custom-jinja2-filters-from-another-file-and-using-flask


@jinja2.pass_context
@jinjablue.app_template_filter()
def make_token(context, value, group):
    ''' Make a keyword string for query parameter dropdown live search '''
    tokstring = ', '.join([group, value._joinedname])
    return tokstring


@jinja2.pass_context
@jinjablue.app_template_filter()
def filtergaltype(context, value):
    ''' Parse plateifu or mangaid into better form '''
    if value == 'plateifu':
        return 'Plate-IFU'
    elif value == 'mangaid':
        return 'MaNGA-ID'


@jinja2.pass_context
@jinjablue.app_template_filter()
def filternsa(context, value):
    ''' Parse plateifu or mangaid into better form '''

    newvalue = value.replace('elpetro_absmag_g_r', 'Abs. g-r').\
        replace('elpetro_absmag_u_r', 'Abs. u-r').\
        replace('elpetro_absmag_i_z', 'Abs. i-z')
    return newvalue


@jinja2.pass_context
@jinjablue.app_template_filter()
def filternsaval(context, value, key):
    ''' Parse plateifu or mangaid into better form '''

    if type(value) == list:
        newvalue = ', '.join([str(np.round(v, 4)) for v in value])
    elif isinstance(value, (float, np.floating)):
        newvalue = np.round(value, 4)
    else:
        newvalue = value

    return newvalue


@jinja2.pass_context
@jinjablue.app_template_filter()
def allclose(context, value, newvalue):
    ''' Do a numpy allclose comparison between the two values '''
    try:
        return np.allclose(float(value), float(newvalue), 1e-7)
    except Exception as e:
        return False


@jinja2.pass_context
@jinjablue.app_template_filter()
def prettyFlag(context, value):
    ''' Pretty print bit mask and flags '''
    name, bit, flags = value
    return '{0}: {1} - {2}'.format(name, bit, ', '.join(flags))


@jinja2.pass_context
@jinjablue.app_template_filter()
def qaclass(context, value):
    ''' Return an alert indicator based on quality flags '''
    name, bit, flags = value
    isgood = ['VALIDFILE'] == flags or [] == flags
    iscrit = 'CRITICAL' in flags
    out = 'success' if isgood else 'danger' if iscrit else 'warning'
    text = 'Good' if isgood else 'DO NOT USE' if iscrit else 'Warning'
    return out, text


@jinja2.pass_context
@jinjablue.app_template_filter()
def targtype(context, value):
    ''' Return the MaNGA target type based on what bit is set '''
    # names = value.get('names', None)
    # namelabel = ', '.join(names)
    # out = namelabel.replace('MNGTRG1', 'Galaxy').replace('MNGTRG2', 'Stellar').replace('MNGTRG3', 'Ancillary')
    out = 'Galaxy' if '1' in value else 'Ancillary' if '3' in value else 'Stellar'
    return out


@jinja2.pass_context
@jinjablue.app_template_filter()
def split(context, value, delim=None):
    '''Split a string based on a delimiter'''
    if not delim:
        delim = ' '
    return value.split(delim) if value else None


@jinja2.pass_context
@jinjablue.app_template_filter()
def striprelease(context, value):
    '''Strip and trim and lowercase a release string'''
    value = value.strip().replace('-', '').lower()
    return value
