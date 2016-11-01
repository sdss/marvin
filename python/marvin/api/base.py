#!/usr/bin/env python
# encoding: utf-8

'''
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-02-17 17:46:57
    Last Modified On: 2016-02-17 17:46:57 by Brian

'''
from __future__ import print_function
from __future__ import division
from brain.api.base import BrainBaseView
from marvin import config


class BaseView(BrainBaseView):
    '''Super Class for all API Views to handle all global API items of interest'''

    def add_config(self):
        utahconfig = {'utahconfig': {'mode': config.mode, 'release': config.release}}
        self.update_results(utahconfig)
