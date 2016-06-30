#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-26 10:42:12
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-26 10:42:12 by Brian Cherinka
    Last Modified On: 2016-04-26 10:42:12 by Brian

'''
from __future__ import print_function, division
from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from brain.api.base import processRequest
from marvin.core import MarvinError
from marvin.tools.query import doQuery, Query
from marvin.tools.query.forms import MarvinForm
import os
import random
import json

search = Blueprint("search_page", __name__)


def getRandomQuery():
    ''' Return a random query from this list '''
    samples = ['nsa.z < 0.02 and ifu.name = 19*', 'cube.plate < 8000']
    q = random.choice(samples)
    return q


class Search(FlaskView):
    route_base = '/search'

    def __init__(self):
        ''' Initialize the route '''
        self.search = {}
        self.search['title'] = 'Marvin | Search'
        self.search['page'] = 'marvin-search'
        self.search['error'] = None
        self.mf = MarvinForm()

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        self.search['results'] = None
        self.search['errmsg'] = None
        self.search['filter'] = None

    @route('/', methods=['GET', 'POST'])
    def index(self):

        # Attempt to retrieve search parameters
        form = processRequest(request=request, raw=True)
        self.search['formparams'] = form

        # set the marvin form
        searchform = self.mf.SearchForm(form)
        q = Query()
        allparams = q.get_available_params()
        searchform.returnparams.choices = [(k.lower(), k) for k in allparams]

        # Add the forms
        self.search['searchform'] = searchform
        self.search['placeholder'] = getRandomQuery()

        # If form parameters then try to search
        if form:
            print('searchform', form)
            self.search.update({'results': None, 'errmsg': None})
            searchvalue = form['searchbox']
            # Get the returnparams from the dropdown select
            returnparams = form.getlist('returnparams', None)
            # Get the returnparams from the autocomplete input
            parambox = form.get('parambox', None)
            parambox = parambox.split(',')[:-1] if parambox else None
            # Select the one that is not none
            returnparams = returnparams if returnparams else parambox if parambox else None
            current_session.update({'searchvalue': searchvalue, 'returnparams': returnparams})
            # if main form passes validation then do search
            if searchform.validate():
                # try the query
                try:
                    q, res = doQuery(searchfilter=searchvalue, returnparams=returnparams)
                except MarvinError as e:
                    self.search['errmsg'] = 'Could not perform query: {0}'.format(e)
                else:
                    self.search['filter'] = q.strfilter
                    self.search['count'] = res.totalcount
                    if res.count > 0:
                        cols = res.mapColumnsToParams()
                        rows = res.getDictOf(format_type='listdict')
                        output = {'total': res.totalcount, 'rows': rows, 'columns': cols}
                    else:
                        output = None
                    self.search['results'] = output
                    self.search['reslen'] = len(res.results)

        return render_template('search.html', **self.search)

    @route('/getparams/', methods=['GET', 'POST'], endpoint='getparams')
    def getparams(self):
        ''' Retrieves the list of query parameters for Bloodhound Typeahead

        '''
        q = Query()
        allparams = q.get_available_params()
        output = json.dumps(allparams)
        return output

Search.register(search)

