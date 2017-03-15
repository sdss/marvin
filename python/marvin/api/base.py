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
from brain.api.general import BrainGeneralRequestsView
from marvin import config
from marvin.api import ArgValidator
import json

arg_validate = ArgValidator(urlmap=config.urlmap)


class BaseView(BrainBaseView):
    '''Super Class for all API Views to handle all global API items of interest'''

    def add_config(self):
        utahconfig = {'utahconfig': {'mode': config.mode, 'release': config.release}}
        self.update_results(utahconfig)

    def _pop_args(self, kwargs, arglist=None):
        if arglist:
            arglist = [arglist] if not isinstance(arglist, (list, tuple)) else arglist

            for item in arglist:
                tmp = kwargs.pop(item, None)

        return kwargs

    def before_request(self, *args, **kwargs):
        super(BaseView, self).before_request(*args, **kwargs)

        # try to get a local version of the urlmap for the arg_validator
        if not arg_validate.urlmap:
            bv = BrainGeneralRequestsView()
            resp = bv.buildRouteMap()
            config.urlmap = json.loads(resp.get_data())['urlmap']
            arg_validate.urlmap = config.urlmap

