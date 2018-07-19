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
from brain.utils.general import build_routemap
from marvin import config
from marvin.api import ArgValidator, set_api_decorators
from flask import current_app


arg_validate = ArgValidator(urlmap=None)


class BaseView(BrainBaseView):
    '''Super Class for all API Views to handle all global API items of interest'''
    decorators = set_api_decorators()

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

        # try to get a local version of the urlmap for the arg_validator
        if not arg_validate.urlmap:
            urlmap = build_routemap(current_app)
            config.urlmap = urlmap
            arg_validate.urlmap = urlmap

        super(BaseView, self).before_request(*args, **kwargs)
