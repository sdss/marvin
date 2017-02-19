# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-12-08 14:24:58
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-18 20:55:19

from __future__ import print_function, division, absolute_import
from flask_classy import FlaskView
from flask import request
from marvin.web.web_utils import parseSession


class BaseWebView(FlaskView):
    ''' This is the Base Web View for all pages '''

    def __init__(self, page):
        self.base = {}
        self.base['intro'] = 'Welcome to Marvin!'
        self.update_title(page)
        self._endpoint = self._release = None
        self._drpver = self._dapver = None

        # self.galaxy['title'] = 'Marvin | Galaxy'
        # self.galaxy['page'] = 'marvin-galaxy'
        # self.galaxy['error'] = None
        # self.galaxy['specmsg'] = None
        # self.galaxy['mapmsg'] = None
        # self.galaxy['toggleon'] = 'false'
        # self.galaxy['nsamsg'] = None
        # self.galaxy['nsachoices'] = {'1': {'y': 'z', 'x': 'sersic_logmass', 'xtitle': 'Stellar Mass',
        #                                    'ytitle': 'Redshift', 'title': 'Redshift vs Stellar Mass'},
        #                              '2': {'y': 'elpetro_mag_g_r', 'x': 'sersic_absmag_r', 'xtitle': 'AbsMag_r',
        #                                    'ytitle': 'g-r', 'title': 'g-r vs Abs. Mag r'}
        #                              }
        # self.galaxy['nsaplotcols'] = ['z', 'sersic_logmass', 'sersic_n', 'sersic_absmag_r', 'elpetro_mag_g_r',
        #                               'elpetro_th50_r', 'elpetro_mag_u_r', 'elpetro_mag_i_z', 'elpetro_ba',
        #                               'elpetro_phi', 'sersic_mtol_r', 'elpetro_th90_r']
        # self.random['title'] = 'Marvin | Random'
        # self.random['page'] = 'marvin-random'
        # self.random['error'] = None

        # self.base['title'] = 'Marvin'
        # self.base['intro'] = 'Welcome to Marvin!'
        # self.base['page'] = 'marvin-main'

        # self.search['title'] = 'Marvin | Search'
        # self.search['page'] = 'marvin-search'
        # self.search['error'] = None
        # # self.mf = MarvinForm()

        # self.user['title'] = 'Marvin | User'
        # self.user['page'] = 'marvin-user'
        # self.user['error'] = None


    def before_request(self, *args, **kwargs):
        ''' this runs before every single request '''
        self.base['error'] = None

        # self.galaxy['cube'] = None
        # self.galaxy['image'] = ''
        # self.galaxy['spectra'] = 'null'
        # self.galaxy['maps'] = None
        # self.search['results'] = None
        # self.search['errmsg'] = None
        # self.search['filter'] = None

        self._endpoint = request.endpoint
        self._drpver, self._dapver, self._release = parseSession()

    def after_request(self, name, response):
        ''' this runs after every single request '''

        return response

    def update_title(self, page):
        ''' Update the title and page '''
        self.base['title'] = page.title().split('-')[0] if 'main' in page \
            else page.title().replace('-', ' | ')
        self.base['page'] = page

    def reset_dict(self, mydict, exclude=None):
        ''' resets the page dictionary '''
        if exclude:
            exclude = exclude if isinstance(exclude, list) else [exclude]
        diffkeys = set(mydict) - set(self.base)
        for key, val in mydict.items():
            if key in diffkeys and (exclude and key not in exclude):
                mydict[key] = '' if isinstance(val, str) else None
