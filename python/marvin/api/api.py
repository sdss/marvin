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

<<<<<<< 76cb53c407a8eee40eed16280f67e7ffb8b3ae12
configkeys = ['mplver', 'drver']
=======
configkeys = ['mplver', 'drpver', 'dapver', 'session_id']
>>>>>>> testing login


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
            Relative url path of the API call you want to make

        params (dict):
            dictionary of parameters you are passing into the API function call

        request_type (str):
            the method type of the API call, can be either "get" or "post" (default: post)

    Returns:
        results (dict):
            The results of the API call

    Examples:
        >>>
        >>>
        >>>

    '''

    def _loadConfigParams(self):
        """Load the local configuration into a parameters dictionary to be sent with the request"""

        from marvin import config

        if self.params:
            for k in configkeys:
                if k not in self.params or self.params[k] is None:
                    self.params[k] = config.__getattribute__(k)
        else:
            self.params = {k: config.__getattribute__(k) for k in configkeys}
