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
from marvin.api import ArgValidator

arg_validate = ArgValidator(urlmap=config.urlmap)


class BaseView(BrainBaseView):
    '''Super Class for all API Views to handle all global API items of interest'''

    def add_config(self):
        utahconfig = {'utahconfig': {'mode': config.mode, 'release': config.release}}
        self.update_results(utahconfig)

    def pop_args(self, kwargs, arglist=None):
        ''' Pop a list of arguments out of the arg/kwargs '''
        if arglist:
            arglist = [arglist] if not isinstance(arglist, (list, tuple)) else arglist

            for item in arglist:
                tmp = kwargs.pop(item, None)

        return kwargs
