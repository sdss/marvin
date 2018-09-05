#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-26 09:20:35
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-26 09:20:35 by Brian Cherinka
    Last Modified On: 2016-04-26 09:20:35 by Brian

'''
from __future__ import print_function
from __future__ import division
from brain.api.api import BrainInteraction
from marvin import config

configkeys = ['release', 'session_id', 'compression']


class Interaction(BrainInteraction):
    ''' Marvins Interaction class, subclassed from Brain

    This is the main class to make calls to the Marvin API.  Instantaiate
    the Interaction object with a URL to make the call.

    GET requests can be made without passing parameters.
    POST requests require parameters to be passed.

    A successful call results in a HTTP status code of 200.  Failures result
    in some other HTTP status code.  Results from the successful call are stored
    in an sttribute called results, as a dictionary.  Any data requested is
    stored as a key inside results called "data"

    Parameters:
        route (str):
            Required.  Relative url path of the API call you want to make
        params (dict):
            dictionary of parameters you are passing into the API function call
        request_type (str):
            the method type of the API call, can be either "get" or "post" (default: post)
        auth (str):
            the authentication method used for the API.  Currently set as default to use
            netrc authentication.
        timeout (float|tuple):
            A float or tuple of floats indicating the request timeout limit in seconds.
            If the server has not sent a respsonse by the time limit, an exception is raised.
            The default timeout is set to 5 minutes.
            See http://docs.python-requests.org/en/master/user/advanced/#timeouts
        headers (dict):
            A custom header to send with the request
        stream (bool):
            If True, iterates over the response data.  Default is False.  When set, avoids reading the
            content all at once into memory for large responses.
            See `request streaming <http://docs.python-requests.org/en/master/user/advanced/#streaming-requests>`_
        datastream (bool):
            If True, expects the response content to be streamed back using a Python generator.
            All matters when Marvin Query return_all is True.

    Returns:
        results (dict):
            The **Response JSON object** from the API call.  If the API is successful, the json data is extracted
            from the response and stored in this dictionary.  See :ref:`marvin-api-routes` for a
            description of the contents of results in each route.

    Examples:
        >>> from marvin import config
        >>> config.mode = 'remote'
        >>>
        >>> # import the Marvin Interaction class
        >>> from marvin.api.api import Interaction
        >>>
        >>> # get and format an API url to retrieve basic Cube properties
        >>> plateifu = '7443-12701'
        >>> url = config.urlmap['api']['getCube']['url']
        >>>
        >>> # create and send the request, and retrieve a response
        >>> response = Interaction(url.format(name=plateifu))
        >>>
        >>> # check your response's status code
        >>> print(response.status_code)
        >>> 200
        >>>
        >>> # get the data in your response
        >>> data = response.getData()
        >>> print(data)
    '''

    def _loadConfigParams(self):
        """Load the local configuration into a parameters dictionary to be sent with the request"""

        if self.params:
            for k in configkeys:
                if k not in self.params or self.params[k] is None:
                    self.params[k] = config.__getattribute__(k)
        else:
            self.params = {k: config.__getattribute__(k) for k in configkeys}

    def setAuth(self, authtype=None):
        ''' Set the authorization '''

        release = self.params['release'] if self.params and 'release' in self.params else config.release

        if (config.access == 'collab' and 'DR' in release) or config.access == 'public':
            authtype = None
        else:
            assert authtype is not None, 'Must have an authorization type set for collab access to MPLs!'

        super(Interaction, self).setAuth(authtype=authtype)
