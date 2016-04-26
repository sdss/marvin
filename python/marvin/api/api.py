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

configkeys = ['mplver', 'drpver', 'dapver']


class Interaction(BrainInteraction):
    ''' Marvins Interaction class, subclassed from Brain '''

    def _loadConfigParams(self):
        """Load the local configuration into a parameters dictionary to be sent with the request"""
        if self.params:
            for k in configkeys:
                self.params[k] = config.__getattribute__(k)
        else:
            self.params = {k: config.__getattribute__(k) for k in configkeys}


