#!/usr/bin/env python
# encoding: utf-8
"""

general.py

Licensed under a 3-clause BSD license.

Revision history:
    17 Feb 2016 J. Sánchez-Gallego
      Initial version
    18 Feb 2016 B. Cherinka
        Added buildRouteMap API call

"""

from __future__ import division
from __future__ import print_function
from flask.ext.classy import route
from marvin.api.base import BaseView
from marvin.utils.general import mangaid2plateifu as mangaid2plateifu
#from marvin.api.cube import api
from flask import url_for, current_app
import json
import urllib


#apiGeneral = Blueprint("apiGeneral", __name__)


class GeneralRequestsView(BaseView):
    """A collection of requests for generic purposes."""

    route_base = '/general/'

    @route('/mangaid2plateifu/<mangaid>', endpoint='mangaid2plateifu')
    def mangaid2plateifu(self, mangaid):

        try:
            plateifu = mangaid2plateifu(mangaid, mode='db')
            self.results['data'] = plateifu
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = ('manga2plateifu failed with error: {0}'.format(str(ee)))

        return json.dumps(self.results)

    @route('/getroutemap', endpoint='getroutemap')
    def buildRouteMap(self):
        ''' Build the URL route map for all routes in the Flask app.

            Returns in self.results a key 'urlmap' of dictionary of routes.
            Syntax:  {blueprint: {endpoint: {'methods':x, 'url':x} }
            E.g. getSpectrum method
            urlmap = {'api': {'getspectra': {'methods':['GET','POST'], 'url': '/api/cubes/{name}/spectra/{path}'} } }

            urls can now easily handle variable replacement in real code; MUST use keyword substitution
            E.g.
            print urlmap['api']['getspectra']['url'].format(name='1-209232',path='x=10/y=5')
            returns '/api/cubes/1-209232/spectra/x=10/y=5'
        '''

        output = {}
        for rule in current_app.url_map.iter_rules():
            # get options
            options = {}
            for arg in rule.arguments:
                options[arg] = '[{0}]'.format(arg)
            # get endpoint
            fullendpoint = rule.endpoint
            esplit = fullendpoint.split('.')
            grp, endpoint = esplit[0], None if len(esplit) == 1 else esplit[1]
            output.setdefault(grp, {}).update({endpoint: {}})
            # get methods
            methods = ','.join(rule.methods)
            output[grp][endpoint]['methods'] = methods
            # build url
            rawurl = url_for(fullendpoint, **options)
            url = urllib.unquote(rawurl).replace('[', '{').replace(']', '}').strip('/')
            output[grp][endpoint]['url'] = url

        res = {'urlmap': output}
        self.update_results(res)
        return json.dumps(self.results)


#GeneralRequestsView.register(apiGeneral)
