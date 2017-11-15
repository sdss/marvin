#!/usr/bin/env python
# encoding: utf-8
#
# plotting.py
#
# Created by José Sánchez-Gallego on 8 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


__ALL__ = ('get_dap_maplist', 'get_default_mapset', 'get_default_plot_params')


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


def get_default_mapset(dapver=None):
    ''' Returns a list of six default maps for display '''

    dapdefaults = {
        # 6 defaults
        # '1.1.1': ['emline_gflux:oiid_3728', 'emline_gflux:hb_4862', 'emline_gflux:oiii_5008',
        #           'emline_gflux:ha_6564', 'emline_gflux:nii_6585', 'emline_gflux:sii_6718'],
        # '2.0.2': ['emline_gflux:oiid_3728', 'emline_gflux:hb_4862', 'emline_gflux:oiii_5008',
        #           'emline_gflux:ha_6564', 'emline_gflux:nii_6585', 'emline_gflux:sii_6718']
        # 3 defaults
        '1.1.1': ['stellar_vel', 'emline_gflux:ha_6564', 'specindex:d4000'],
        '2.0.2': ['stellar_vel', 'emline_gflux:ha_6564', 'specindex:d4000']
    }

    return dapdefaults[dapver] if dapver in dapdefaults else []


def get_default_plot_params(dapver=None):
    """Returns default map plotting parameters."""

    bitmasks = {'1.1.1': ['DONOTUSE'],
                '2.0.2': ['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                '2.1': ['NOCOV', 'UNRELIABLE', 'DONOTUSE']  # TODO update to 2.1.0
                }

    mpl5 = {'default': {'bitmasks': bitmasks['2.0.2'],
                        'cmap': 'linearlab',
                        'percentile_clip': [5, 95],
                        'symmetric': False,
                        'snr_min': 1},
            'vel': {'bitmasks': bitmasks['2.0.2'],
                    'cmap': 'RdBu_r',
                    'percentile_clip': [10, 90],
                    'symmetric': True,
                    'snr_min': None},
            'sigma': {'bitmasks': bitmasks['2.0.2'],
                      'cmap': 'inferno',
                      'percentile_clip': [10, 90],
                      'symmetric': False,
                      'snr_min': 1}}

    plot_defaults = {
        '1.1.1': {'default': {'bitmasks': bitmasks['1.1.1'],
                              'cmap': 'linearlab',
                              'percentile_clip': [5, 95],
                              'symmetric': False,
                              'snr_min': 1},
                  'vel': {'bitmasks': bitmasks['1.1.1'],
                          'cmap': 'RdBu_r',
                          'percentile_clip': [10, 90],
                          'symmetric': True,
                          'snr_min': None},
                  'sigma': {'bitmasks': bitmasks['1.1.1'],
                            'cmap': 'inferno',
                            'percentile_clip': [10, 90],
                            'symmetric': False,
                            'snr_min': 1}},
        '2.0.2': mpl5,
        '2.1': mpl5  # TODO Update to 2.1.0
    }

    return plot_defaults[dapver] if dapver in plot_defaults else {}
