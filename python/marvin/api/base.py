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
from flask.ext.classy import FlaskView


class BaseView(FlaskView):
    ''' Super Clase for all API Views to handle all global API things of interest '''

    def __init__(self):
        self.reset_results()

    def reset_results(self):
        ''' Resets results to return from API as JSON. '''
        self.results = {'data': None, 'status': -1, 'error': None}

    def update_results(self, newresults):
        ''' Add to or Update the results dictionary '''
        self.results.update(newresults)

    def reset_status(self):
        ''' Resets the status to -1 '''
        self.results['status'] = -1



