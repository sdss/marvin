#!/usr/bin/env python
# encoding: utf-8
#
# plotting.py
#
# Created by José Sánchez-Gallego on 8 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from marvin import config


__ALL__ = ('get_dap_maplist', 'get_default_mapset', 'get_default_plot_params')


DAPKEYS = [config.lookUpVersions(k)[1] for k in config._mpldict.keys()
           if config.lookUpVersions(k)[1] and 'v' not in config.lookUpVersions(k)[1]]


def get_dap_maplist(dapver=None, web=None):
    ''' Returns a list of all possible maps for dapver '''

    from . import datamodel

    dapdm = datamodel[dapver]
    daplist = []

    for p in dapdm.properties:
        if p.channel:
            if web:
                daplist.append('{0}:{1}'.format(p.name, p.channel))
            else:
                daplist.append('{0}_{1}'.format(p.name, p.channel))
        else:
            daplist.append(p.name)

    return daplist


def get_default_mapset(dapver=None, defaults=None):
    ''' Returns a list of six default maps for display '''

    if not defaults:
        defaults = ['stellar_vel', 'emline_gflux:ha_6564', 'specindex:d4000']

    dapdefaults = {
        '1.1.1': defaults,
        '2.0.2': defaults,
        '2.1.3': defaults,
        '2.2.1': defaults,
    }

    assert set(DAPKEYS).issubset(set(dapdefaults.keys())), 'DAP default_mapset versions must be up-to-date with MPL versions'

    return dapdefaults[dapver] if dapver in dapdefaults else []


def set_base_params(bitmasks):
    ''' Set the base plotting param defaults '''

    return {'default': {'bitmasks': bitmasks,
                        'cmap': 'linearlab',
                        'percentile_clip': [5, 95],
                        'symmetric': False,
                        'snr_min': 1},
            'vel': {'bitmasks': bitmasks,
                    'cmap': 'RdBu_r',
                    'percentile_clip': [10, 90],
                    'symmetric': True,
                    'snr_min': None},
            'sigma': {'bitmasks': bitmasks,
                      'cmap': 'inferno',
                      'percentile_clip': [10, 90],
                      'symmetric': False,
                      'snr_min': 1}}


def get_default_plot_params(dapver=None):
    """Returns default map plotting parameters."""

    bitmasks = {'1.1.1': ['DONOTUSE'],
                '2.0.2': ['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                '2.1.3': ['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                '2.2.1': ['NOCOV', 'UNRELIABLE', 'DONOTUSE']
                }

    mpl4 = set_base_params(bitmasks['1.1.1'])
    mpl5 = set_base_params(bitmasks['2.0.2'])
    mpl6 = set_base_params(bitmasks['2.1.3'])
    mpl7 = set_base_params(bitmasks['2.2.1'])

    plot_defaults = {
        '1.1.1': mpl4,
        '2.0.2': mpl5,
        '2.1.3': mpl6,
        '2.2.1': mpl7
    }

    assert set(DAPKEYS).issubset(set(plot_defaults.keys())), \
        'DAP default_plot_param versions must be up-to-date with MPL versions'

    return plot_defaults[dapver] if dapver in plot_defaults else {}
