import json
from flask_classy import route
from flask import request, jsonify
from marvin.tools.query import doQuery, Query
from marvin.core.exceptions import MarvinError
from marvin.api import parse_params
from marvin.api.base import BaseView
from marvin.utils.db import get_traceback


def _getCubes(searchfilter, params=None, rettype=None, start=None, end=None,
              limit=None, sort=None, order=None, release=None):
    """Run query locally at Utah."""

    if release is None:
        release = parse_params(request)

    try:
        q, r = doQuery(searchfilter=searchfilter, returnparams=params, release=release,
                       mode='local', returntype=rettype, limit=limit, order=order, sort=sort)
    except Exception as e:
        raise MarvinError('Query failed with {0}: {1}'.format(e.__class__.__name__, e))

    results = r.results

    # get a subset
    chunk = None
    if start:
        chunk = int(end)-int(start)
        results = r.getSubset(int(start), limit=chunk)
    chunk = limit if not chunk else limit
    runtime = {'days': q.runtime.days, 'seconds': q.runtime.seconds, 'microseconds': q.runtime.microseconds}
    output = dict(data=results, query=r.showQuery(), chunk=limit,
                  filter=searchfilter, params=q.params, returnparams=params, runtime=runtime,
                  queryparams_order=q.queryparams_order, count=len(results), totalcount=r.count)
    return output


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    def index(self):
        '''Returns general query info

        .. :quickref: Query; Get general query info

        :query string release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: data message
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/query/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": "this is a query!"
           }

        '''
        self.results['data'] = 'this is a query!'
        return jsonify(self.results)

    @route('/cubes/', methods=['GET', 'POST'], endpoint='querycubes')
    def cube_query(self):
        ''' Performs a remote query

        .. :quickref: Query; Perform a remote query

        :query string release: the release of MaNGA
        :form searchfilter: your string searchfilter expression
        :form params: the list of return parameters
        :form rettype: the string indicating your Marvin Tool conversion object
        :form limit: the limiting number of results to return for large results
        :form sort: a string parameter name to sort on
        :form order: the order of the sort, either ``desc`` or ``asc``
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: dictionary of returned data
        :json list results: the list of results
        :json string query: the raw SQL string of your query
        :json int chunk: the page limit of the results
        :json string filter: the searchfilter used
        :json list returnparams: the list of return parameters
        :json list params: the list of parameters used in the query
        :json list queryparams_order: the list of parameters used in the query
        :json dict runtime: a dictionary of query time (days, minutes, seconds)
        :json int totalcount: the total count of results
        :json int count: the count in the current page of results
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/query/cubes/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5", "searchfilter": "nsa.z<0.1"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "chunk": 100,
              "count": 4,
              "data": [["1-209232",8485,"8485-1901","1901",0.0407447],
                       ["1-209113",8485,"8485-1902","1902",0.0378877],
                       ["1-209191",8485,"8485-12701","12701",0.0234253],
                       ["1-209151",8485,"8485-12702","12702",0.0185246]
                      ],
              "filter": "nsa.z<0.1",
              "params": ["cube.mangaid","cube.plate","cube.plateifu","ifu.name","nsa.z"],
              "query": "SELECT ... FROM ... WHERE ...",
              "queryparams_order": ["mangaid","plate","plateifu","name","z"],
              "returnparams": null,
              "runtime": {"days": 0,"microseconds": 55986,"seconds": 0},
              "totalcount": 4
           }

        '''
        searchfilter = self.results['inconfig'].get('searchfilter', None)
        params = self.results['inconfig'].get('params', None)
        rettype = self.results['inconfig'].get('returntype', None)
        limit = self.results['inconfig'].get('limit', 100)
        sort = self.results['inconfig'].get('sort', None)
        order = self.results['inconfig'].get('order', 'asc')
        release = self.results['inconfig'].get('release', None)

        print('inconfig', self.results['inconfig'])
        print('cube_query', searchfilter, params, limit)
        try:
            res = _getCubes(searchfilter, params=params, rettype=rettype,
                            limit=limit, sort=sort, order=order, release=release)
        except MarvinError as e:
            self.results['error'] = str(e)
            self.results['traceback'] = get_traceback(asstring=True)
        else:
            self.results['status'] = 1
            self.update_results(res)

        return jsonify(self.results)

    @route('/cubes/getsubset/', methods=['GET', 'POST'], endpoint='getsubset')
    def query_getsubset(self):
        ''' Remotely grab a subset of results from a query

        .. :quickref: Query; Grab a subset of results from a remote query

        :query string release: the release of MaNGA
        :form searchfilter: your string searchfilter expression
        :form params: the list of return parameters
        :form rettype: the string indicating your Marvin Tool conversion object
        :form start: the starting page index of results you wish to grab
        :form end: the ending page index of the results you wish to grab
        :form limit: the limiting number of results to return for large results
        :form sort: a string parameter name to sort on
        :form order: the order of the sort, either ``desc`` or ``asc``
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: dictionary of returned data
        :json list results: the list of results
        :json string query: the raw SQL string of your query
        :json int chunk: the page limit of the results
        :json string filter: the searchfilter used
        :json list returnparams: the list of return parameters
        :json list params: the list of parameters used in the query
        :json list queryparams_order: the list of parameters used in the query
        :json dict runtime: a dictionary of query time (days, minutes, seconds)
        :json int totalcount: the total count of results
        :json int count: the count in the current page of results
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/query/cubes/getsubset/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5", "searchfilter": "nsa.z<0.1", "start":10, "end":15},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "chunk": 100,
              "count": 4,
              "data": [["1-209232",8485,"8485-1901","1901",0.0407447],
                       ["1-209113",8485,"8485-1902","1902",0.0378877],
                       ["1-209191",8485,"8485-12701","12701",0.0234253],
                       ["1-209151",8485,"8485-12702","12702",0.0185246]
                      ],
              "filter": "nsa.z<0.1",
              "params": ["cube.mangaid","cube.plate","cube.plateifu","ifu.name","nsa.z"],
              "query": "SELECT ... FROM ... WHERE ...",
              "queryparams_order": ["mangaid","plate","plateifu","name","z"],
              "returnparams": null,
              "runtime": {"days": 0,"microseconds": 55986,"seconds": 0},
              "totalcount": 4
           }

        '''
        searchfilter = self.results['inconfig'].get('searchfilter', None)
        params = self.results['inconfig'].get('params', None)
        start = self.results['inconfig'].get('start', None)
        end = self.results['inconfig'].get('end', None)
        rettype = self.results['inconfig'].get('returntype', None)
        limit = self.results['inconfig'].get('limit', 100)
        sort = self.results['inconfig'].get('sort', None)
        order = self.results['inconfig'].get('order', 'asc')
        release = self.results['inconfig'].get('release', None)

        try:
            res = _getCubes(searchfilter, params=params, start=int(start),
                            end=int(end), rettype=rettype, limit=limit,
                            sort=sort, order=order, release=release)
        except MarvinError as e:
            self.results['error'] = str(e)
            self.results['traceback'] = get_traceback(asstring=True)
        else:
            self.results['status'] = 1
            self.update_results(res)

        return jsonify(self.results)

    @route('/getparamslist/', methods=['GET', 'POST'], endpoint='getparams')
    def getparamslist(self):
        ''' Retrieve a list of all available input parameters into the query

        .. :quickref: Query; Get a list of all or "best" queryable parameters

        :query string release: the release of MaNGA
        :form paramdisplay: ``all`` or ``best``, type of parameters to return
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: dictionary of returned data
        :json list params: the list of queryable parameters
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/query/getparamslist/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": ['nsa.z', 'cube.ra', 'cube.dec', ...]
           }

        '''
        paramdisplay = self.results['inconfig'].get('paramdisplay', 'all')
        q = Query(mode='local')
        if paramdisplay == 'all':
            params = q.get_available_params()
        elif paramdisplay == 'best':
            params = q.get_best_params()
        self.results['data'] = params
        self.results['status'] = 1
        output = jsonify(self.results)
        return output

    @route('/cleanup/', methods=['GET', 'POST'], endpoint='cleanupqueries')
    def cleanup(self):
        ''' Clean up idle server-side queries or retrieve the list of them

        Do not use!

        .. :quickref: Query; Send a cleanup command to the server-side database

        :query string release: the release of MaNGA
        :form task: ``clean`` or ``getprocs``, the type of task to run
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: dictionary of returned data
        :json string clean: clean success message
        :json list procs: the list of processes currently running on the db
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/query/cleanup/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": 'clean success'
           }

        '''
        task = self.results['inconfig'].get('task', None)
        if task == 'clean':
            q = Query(mode='local')
            q._cleanUpQueries()
            res = {'status': 1, 'data': 'clean success'}
        elif task == 'getprocs':
            q = Query(mode='local')
            procs = q._getIdleProcesses()
            procs = [{k: v for k, v in y.items()} for y in procs]
            res = {'status': 1, 'data': procs}
        else:
            res = {'status': -1, 'data': None, 'error': 'Task is None or not in [clean, getprocs]'}
        self.update_results(res)
        output = jsonify(self.results)
        return output
