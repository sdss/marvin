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
from flask import Blueprint, render_template, session as current_session, request, jsonify
from flask_classy import route
from brain.api.base import processRequest
from marvin.core.exceptions import MarvinError
from marvin.tools.query import doQuery, Query
from marvin.tools.query.forms import MarvinForm
from marvin.web.controllers import BaseWebView
from marvin.api.base import arg_validate as av
from wtforms import validators, ValidationError
import random

search = Blueprint("search_page", __name__)


def getRandomQuery():
    ''' Return a random query from this list '''
    samples = ['nsa.z < 0.02 and ifu.name = 19*', 'cube.plate < 8000', 'haflux > 25',
               'nsa.sersic_logmass > 9.5 and nsa.sersic_logmass < 11', 'emline_ew_ha_6564 > 3']
    q = random.choice(samples)
    return q


def all_in(fullist):
    ''' search form query parameter form validation '''
    def _all_in(form, field):
        myparams = [f for f in field.data if f]
        outsiders = list(set(myparams) - set(fullist))
        message = '{0} must be a given query parameter'.format(', '.join(outsiders))
        if outsiders:
            raise ValidationError(message)
    return _all_in


class Search(BaseWebView):
    route_base = '/search/'

    def __init__(self):
        ''' Initialize the route '''
        super(Search, self).__init__('marvin-search')
        self.search = self.base.copy()
        self.search['filter'] = None
        self.search['results'] = None
        self.search['errmsg'] = None
        self.mf = MarvinForm()

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        super(Search, self).before_request(*args, **kwargs)
        self.reset_dict(self.search)

    @route('/', methods=['GET', 'POST'])
    def index(self):

        # Attempt to retrieve search parameters
        form = processRequest(request=request)
        self.search['formparams'] = form

        print('search form', form, type(form))
        # set the search form and form validation
        searchform = self.mf.SearchForm(form)
        q = Query(release=self._release)
        # allparams = q.get_available_params()
        bestparams = q.get_best_params()
        searchform.returnparams.choices = [(k.lower(), k) for k in bestparams]
        searchform.parambox.validators = [all_in(bestparams), validators.Optional()]

        # Add the forms
        self.search['searchform'] = searchform
        self.search['placeholder'] = getRandomQuery()

        #from flask import abort
        #abort(500)

        # If form parameters then try to do a search
        if form:
            self.search.update({'results': None, 'errmsg': None})

            args = av.manual_parse(self, request, use_params='search')
            print('search args', args)

            # get form parameters
            searchvalue = form['searchbox']  # search filter input
            returnparams = form.getlist('returnparams', type=str)  # dropdown select
            parambox = form.get('parambox', None, type=str)  # from autocomplete
            if parambox:
                parms = parambox.split(',')
                parms = parms if parms[-1].strip() else parms[:-1]
                parambox = parms if parambox else None
            # Select the one that is not none
            returnparams = returnparams if returnparams and not parambox else \
                parambox if parambox and not returnparams else \
                list(set(returnparams) | set(parambox)) if returnparams and parambox else None
            print('return params', returnparams)
            current_session.update({'searchvalue': searchvalue, 'returnparams': returnparams})

            # if main form passes validation then do search
            if searchform.validate():
                # try the query
                try:
                    q, res = doQuery(searchfilter=searchvalue, release=self._release, returnparams=returnparams)
                except MarvinError as e:
                    self.search['errmsg'] = 'Could not perform query: {0}'.format(e)
                else:
                    self.search['filter'] = q.strfilter
                    self.search['count'] = res.totalcount
                    self.search['runtime'] = res.query_runtime.total_seconds()
                    if res.count > 0:
                        cols = res.mapColumnsToParams()
                        rows = res.getDictOf(format_type='listdict')
                        output = {'total': res.totalcount, 'rows': rows, 'columns': cols}
                    else:
                        output = None
                    self.search['results'] = output
                    self.search['reslen'] = len(res.results)
                    if returnparams:
                        returnparams = [str(r) for r in returnparams]
                    rpstr = 'returnparams={0} <br>'.format(returnparams) if returnparams else ''
                    qstr = ', returnparams=returnparams' if returnparams else ''
                    self.search['querystring'] = ("<html><samp>from marvin import \
                        config<br>from marvin.tools.query import Query<br>config.mode='remote'<br>\
                        filter='{0}'<br> {1}\
                        q = Query(searchfilter=filter{2})<br>\
                        r = q.run()<br></samp></html>".format(searchvalue, rpstr, qstr))

        return render_template('search.html', **self.search)

    @route('/dopost/', methods=['POST'], endpoint='dopost')
    def post(self):
        return 'this is a post'

    @route('/getparams/<paramdisplay>/', methods=['GET', 'POST'], endpoint='getparams')
    def getparams(self, paramdisplay):
        ''' Retrieves the list of query parameters for Bloodhound Typeahead

        '''

        # set the paramdisplay if it is not
        if not paramdisplay:
            paramdisplay = 'all'

        # run query and retrieve parameters
        q = Query(release=self._release)
        if paramdisplay == 'all':
            params = q.get_available_params()
        elif paramdisplay == 'best':
            params = q.get_best_params()
        output = jsonify(params)
        return output

    @route('/webtable/', methods=['GET', 'POST'], endpoint='webtable')
    def webtable(self):
        ''' Do a query for Bootstrap Table interaction in Marvin web '''

        form = processRequest(request=request)
        args = av.manual_parse(self, request, use_params='query')

        #{'sort': u'cube.mangaid', 'task': None, 'end': None, 'searchfilter': None,
        #'paramdisplay': None, 'start': None, 'rettype': None, 'limit': 10, 'offset': 30,
        #'release': u'MPL-4', 'params': None, 'order': u'asc'}

        # remove args
        __tmp__ = args.pop('release', None)
        __tmp__ = args.pop('searchfilter', None)
        limit = args.get('limit')
        offset = args.get('offset')


        # set parameters
        searchvalue = current_session.get('searchvalue', None)
        returnparams = current_session.get('returnparams', None)
        # limit = form.get('limit', 10, type=int)
        # offset = form.get('offset', None, type=int)
        # order = form.get('order', None, type=str)
        # sort = form.get('sort', None, type=str)
        # search = form.get('search', None, type=str)

        # exit if no searchvalue is found
        if not searchvalue:
            output = jsonify({'webtable_error': 'No searchvalue found', 'status': -1})
            return output

        # do query
        q, res = doQuery(searchfilter=searchvalue, release=self._release, returnparams=returnparams, **args)
        # q, res = doQuery(searchfilter=searchvalue, release=self._release,
        #                  limit=limit, order=order, sort=sort, returnparams=returnparams)
        # get subset on a given page
        __results__ = res.getSubset(offset, limit=limit)
        # get keys
        cols = res.mapColumnsToParams()
        # create output
        rows = res.getDictOf(format_type='listdict')
        output = {'total': res.totalcount, 'rows': rows, 'columns': cols}
        output = jsonify(output)
        return output

Search.register(search)

