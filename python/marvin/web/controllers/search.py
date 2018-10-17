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
from flask_classful import route
from brain.api.base import processRequest
from marvin.core.exceptions import MarvinError
from marvin.tools.query import doQuery, Query
from marvin.utils.datamodel.query.forms import MarvinForm
from marvin.web.controllers import BaseWebView
from marvin.api.base import arg_validate as av
from marvin.utils.datamodel.query.base import query_params, bestparams
from wtforms import ValidationError
from marvin.utils.general import getImagesByList
from marvin.web.web_utils import buildImageDict
from marvin.web.extensions import limiter
import random

search = Blueprint("search_page", __name__)


def getRandomQuery():
    ''' Return a random query from this list '''
    samples = ['nsa.z < 0.02', 'cube.plate < 8000', 'haflux > 25',
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
        self.search['returnparams'] = None
        self.mf = MarvinForm()

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        super(Search, self).before_request(*args, **kwargs)
        self.reset_dict(self.search)

    @route('/', methods=['GET', 'POST'])
    @limiter.limit("60/minute")
    def index(self):

        # Attempt to retrieve search parameters
        form = processRequest(request=request)
        self.search['formparams'] = form

        # set the search form and form validation
        searchform = self.mf.SearchForm(form)
        searchform.returnparams.choices = [(k.lower(), k) for k in query_params.list_params()]

        # Add the forms and parameters
        self.search['paramdata'] = query_params
        self.search['guideparams'] = [{'id': p.full, 'optgroup': group.name, 'type': 'double' if p.dtype == 'float' else p.dtype, 'validation': {'step': 'any'}} for group in query_params for p in group]
        self.search['searchform'] = searchform
        self.search['placeholder'] = getRandomQuery()

        # If form parameters then try to do a search
        if form:
            self.search.update({'results': None, 'errmsg': None})

            args = av.manual_parse(self, request, use_params='search')
            # get form parameters
            searchvalue = form['searchbox']  # search filter input
            returnparams = form.getlist('returnparams', type=str)  # dropdown select
            self.search.update({'returnparams': returnparams})
            current_session.update({'searchvalue': searchvalue, 'returnparams': returnparams})

            # if main form passes validation then do search
            if searchform.validate():
                # try the query
                try:
                    q, res = doQuery(search_filter=searchvalue, release=self._release, return_params=returnparams)
                except MarvinError as e:
                    self.search['errmsg'] = 'Could not perform query: {0}'.format(e)
                else:
                    self.search['filter'] = q.search_filter
                    self.search['count'] = res.totalcount
                    self.search['runtime'] = res.query_time.total_seconds()
                    if res.count > 0:
                        cols = res.columns.remote
                        rows = res.getDictOf(format_type='listdict')
                        output = {'total': res.totalcount, 'rows': rows, 'columns': cols, 'limit': None, 'offset': None}
                    else:
                        output = None
                    self.search['results'] = output
                    self.search['reslen'] = len(res.results)
                    if returnparams:
                        returnparams = [str(r) for r in returnparams]
                    rpstr = 'returnparams={0} <br>'.format(returnparams) if returnparams else ''
                    qstr = ', returnparams=returnparams' if returnparams else ''
                    self.search['querystring'] = ("<html><samp>from marvin.tools.query import Query<br>\
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
            paramdisplay = 'best'

        # run query and retrieve parameters
        q = Query(release=self._release)
        if paramdisplay == 'all':
            params = q.get_available_params('all')
        elif paramdisplay == 'best':
            params = bestparams
        output = jsonify(params)
        return output

    @route('/webtable/', methods=['GET', 'POST'], endpoint='webtable')
    def webtable(self):
        ''' Do a query for Bootstrap Table interaction in Marvin web '''

        form = processRequest(request=request)
        args = av.manual_parse(self, request, use_params='query')

        # remove args
        __tmp__ = args.pop('release', None)
        __tmp__ = args.pop('searchfilter', None)
        limit = args.get('limit')
        offset = args.get('offset')
        if 'sort' in args:
            current_session['query_sort'] = args['sort']

        # set parameters
        searchvalue = current_session.get('searchvalue', None)
        returnparams = current_session.get('returnparams', None)

        # exit if no searchvalue is found
        if not searchvalue:
            output = jsonify({'errmsg': 'No searchvalue found', 'status': -1})
            return output

        # this is to fix the brokeness with sorting on a table column using remote names
        #print('rp', returnparams, args)

        # do query
        try:
            q, res = doQuery(search_filter=searchvalue, release=self._release, return_params=returnparams, **args)
        except Exception as e:
            errmsg = 'Error generating webtable: {0}'.format(e)
            output = jsonify({'status': -1, 'errmsg': errmsg})
            return output

        # get subset on a given page
        try:
            __results__ = res.getSubset(offset, limit=limit)
        except Exception as e:
            errmsg = 'Error getting table page: {0}'.format(e)
            output = jsonify({'status': -1, 'errmsg': errmsg})
            return output
        else:
            # get keys
            cols = res.columns.remote
            # create output
            rows = res.getDictOf(format_type='listdict')
            output = {'total': res.totalcount, 'rows': rows, 'columns': cols, 'limit': limit, 'offset': offset}
            output = jsonify(output)
            return output

    @route('/postage/', methods=['GET', 'POST'], defaults={'page': 1}, endpoint='postage')
    @route('/postage/<page>/', methods=['GET', 'POST'], endpoint='postage')
    def postagestamp(self, page):
        ''' Get the postage stamps for a set of query results in the web '''

        postage = {}
        postage['error'] = None
        postage['images'] = None

        pagesize = 16  # number of rows (images) in a page
        pagenum = int(page)  # current page number
        searchvalue = current_session.get('searchvalue', None)
        if not searchvalue:
            postage['error'] = 'No query found! Cannot generate images without a query.  Go to the Query Page!'
            return render_template('postage.html', **postage)

        sort = current_session.get('query_sort', 'cube.mangaid')
        offset = (pagesize * pagenum) - pagesize
        q, res = doQuery(search_filter=searchvalue, release=self._release, sort=sort, limit=10000)
        plateifus = res.getListOf('plateifu')
        # if a dap query, grab the unique galaxies
        if q._check_query('dap'):
            plateifus = list(set(plateifus))

        # only grab subset if more than 16 galaxies
        if len(plateifus) > pagesize:
            plateifus = plateifus[offset:offset + pagesize]

        # get images
        imfiles = None
        try:
            imfiles = getImagesByList(plateifus, as_url=True, mode='local', release=self._release)
        except MarvinError as e:
            postage['error'] = 'Error: could not get images: {0}'.format(e)
        else:
            images = buildImageDict(imfiles)

        # if image grab failed, make placeholders
        if not imfiles:
            images = buildImageDict(imfiles, test=True, num=pagesize)

        # Compute page stats
        totalpages = int(res.totalcount // pagesize) + int(res.totalcount % pagesize != 0)
        page = {'size': pagesize, 'active': int(page), 'total': totalpages, 'count': res.totalcount}

        postage['page'] = page
        postage['images'] = images
        return render_template('postage.html', **postage)


Search.register(search)

