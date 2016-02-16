from __future__ import print_function
import os
import requests
from marvin import config

configkeys = ['mode', 'mplver', 'drpver', 'dapver']


class Interaction(object):
    ''' This class defines convenience wrappers for the Marvin RESTful API '''

    def __init__(self, route, params=None, request_type='get'):
        self.results = None
        self.route = route
        self.url = os.path.join(config.sasurl, route)
        self.params = params
        self.statuscodes = {200: 'Ok', 401: 'Authentication Required', 404: 'URL Not Found', 500: 'Internal Server Error', 405: 'Method Not Allowed', 400: 'Bad Request'}
        self._sendRequest(request_type)

    def _checkResponse(self, response):
        if response.status_code == 200:
            try:
                self.results = response.json()
            except Exception as e:
                self.results = response.text
                raise RuntimeError('Response not in JSON format. {0} {1}'.format(e, self.results))
        else:
            errmsg = 'Error accessing {0}: {1}'.format(response.url, self.statuscodes[response.status_code])
            self.results = {'http status code': response.status_code, 'message': errmsg}

    def _sendRequest(self, request_type):
        assert request_type in ['get', 'post'], 'Valid request types are "get" and "post".'
        if request_type == 'get':
            r = requests.get(self.url, params=self.params)
        elif request_type == 'post':
            r = requests.post(self.url, data=self.params)

        self._checkResponse(r)
        self._preloadResults()

    def getData(self, astype=None):
        data = self.results['data'] if 'data' in self.results else None

        if astype and data:
            try:
                return astype(data)
            except Exception as e:
                raise Exception('Failed: {0}, {1}'.format(e, data))
        else:
            return data

    def _preloadResults(self):
        for key in configkeys:
            self.results[key] = config.__getattribute__(key)

    def checkConfig(self):
        return {k: self.results[k] for k in configkeys}
