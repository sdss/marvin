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
from marvin.tools.query import doQuery
from marvin.tools.query.forms import MarvinForm
import os

search = Blueprint("search_page", __name__)


class Search(FlaskView):
    route_base = '/search'

    def __init__(self):
        ''' Initialize the route '''
        self.search = {}
        self.search['title'] = 'Marvin | Search'
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
        form = processRequest(request=request)
        self.search['formparams'] = form

        # set the marvin form
        mainform = self.mf.MainForm(**form)
        self.search['searchform'] = mainform

        # If form parameters then try to search
        if form:
            self.search.update({'results': None, 'errmsg': None})
            searchvalue = form['searchbox']
            current_session.update({'searchvalue': searchvalue})
            # if main form passes validation then do search
            if mainform.validate():
                # try the query
                try:
                    q, res = doQuery(searchfilter=searchvalue)
                except MarvinError as e:
                    self.search['errmsg'] = 'Could not perform query: {0}'.format(e)
                else:
                    self.search['filter'] = q.strfilter
                    self.search['count'] = res.count
                    if res.count > 0:
                        cols = res.mapColumnsToParams()
                        rows = res.getDictOf(format_type='listdict')
                        output = {'total': res.count, 'rows': rows, 'columns': cols}
                    else:
                        output = None
                    self.search['results'] = output
                    self.search['reslen'] = len(res.results)

        return render_template('search.html', **self.search)


Search.register(search)

