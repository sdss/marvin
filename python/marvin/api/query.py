from flask_classful import route
from flask import request, jsonify, Response, current_app, redirect, url_for
from marvin.tools.query import doQuery, Query
from marvin.core.exceptions import MarvinError
from marvin.api.base import BaseView, arg_validate as av
from marvin.utils.db import get_traceback
from marvin.utils.datamodel.query.base import bestparams
from marvin.web.extensions import limiter
import json


def _run_query(searchfilter, **kwargs):
    ''' Run the query and return the query and results '''

    release = kwargs.pop('release', None)
    kwargs['returnparams'] = kwargs.pop('params', None)
    kwargs['returntype'] = kwargs.pop('rettype', None)

    try:
        q, r = doQuery(searchfilter=searchfilter, release=release, **kwargs)
    except Exception as e:
        raise MarvinError('Query failed with {0}: {1}'.format(e.__class__.__name__, e))
    else:
        return q, r


def _get_runtime(query):
    ''' Retrive a dictionary of the runtime to pass back in JSON '''
    runtime = {'days': query.runtime.days, 'seconds': query.runtime.seconds, 'microseconds': query.runtime.microseconds}
    return runtime


def _getCubes(searchfilter, **kwargs):
    """Run query locally at Utah and format the output into the full JSON """

    # run the query
    q, r = _run_query(searchfilter, **kwargs)

    # get the subset keywords
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    limit = kwargs.get('limit', None)
    params = kwargs.get('params', None)

    # retrieve a subset
    chunk = None
    if start:
        chunk = int(end) - int(start)
        # results = r.getSubset(int(start), limit=chunk)
    chunk = limit if not chunk else limit

    # get results
    results = r.results

    # set up the output
    output = dict(data=results, query=r.showQuery(), chunk=limit,
                  filter=searchfilter, params=q.params, returnparams=params, runtime=_get_runtime(q),
                  queryparams_order=q.queryparams_order, count=len(results), totalcount=r.totalcount)
    return output


def _get_column(results, colname, format_type=None):
    ''' Gets a column from a Query '''

    column = None
    if format_type == 'list':
        try:
            column = results.getListOf(colname)
        except MarvinError as e:
            raise MarvinError('Cannot get list for column {0}: {1}'.format(colname, e))
    elif format_type in ['listdict', 'dictlist']:
        try:
            if colname == 'None':
                column = results.getDictOf(format_type=format_type)
            else:
                column = results.getDictOf(colname, format_type=format_type)
        except MarvinError as e:
            raise MarvinError('Cannot get dictionary for column {0}: {1}'.format(colname, e))

    return column


class QueryView(BaseView):
    """Class describing API calls related to queries."""
    decorators = [limiter.limit("60/minute")]

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
        self.results['status'] = 1
        return jsonify(self.results)

    @route('/cubes/', methods=['GET', 'POST'], endpoint='querycubes')
    @av.check_args(use_params='query', required='searchfilter')
    def cube_query(self, args):
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
        searchfilter = args.pop('searchfilter', None)

        try:
            res = _getCubes(searchfilter, **args)
        except MarvinError as e:
            self.results['error'] = str(e)
            self.results['traceback'] = get_traceback(asstring=True)
        else:
            self.results['status'] = 1
            self.update_results(res)

        # this needs to be json.dumps until sas-vm at Utah updates to 2.7.11
        return Response(json.dumps(self.results), mimetype='application/json')

    @route('/cubes/columns/', defaults={'colname': None}, methods=['GET', 'POST'], endpoint='getcolumn')
    @route('/cubes/columns/<colname>/', methods=['GET', 'POST'], endpoint='getcolumn')
    @av.check_args(use_params='query', required='searchfilter')
    def query_allcolumn(self, args, colname):
        ''' Retrieves the entire result set for a single column

        .. :quickref: Query; Retrieves the entire result set for a single column

        :query string release: the release of MaNGA
        :form searchfilter: your string searchfilter expression
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: dictionary of returned data
        :json list column: the list of results for the specified column
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/query/cubes/columns/plateifu/ HTTP/1.1
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
              "data": ["8485-1901", "8485-1902", "8485-12701", "7443-12701", "8485-12702"],
           }

        '''
        searchfilter = args.pop('searchfilter', None)
        format_type = args.pop('format_type', 'list')

        try:
            query, results = _run_query(searchfilter, **args)
        except MarvinError as e:
            self.results['error'] = str(e)
            self.results['traceback'] = get_traceback(asstring=True)
        else:
            try:
                column = _get_column(results, colname, format_type=format_type)
            except MarvinError as e:
                self.results['error'] = str(e)
                self.results['traceback'] = get_traceback(asstring=True)
            else:
                self.results['status'] = 1
                self.results['data'] = column
                self.results['runtime'] = _get_runtime(query)

        return Response(json.dumps(self.results), mimetype='application/json')

    @route('/cubes/getsubset/', methods=['GET', 'POST'], endpoint='getsubset')
    @av.check_args(use_params='query', required=['searchfilter', 'start', 'end'])
    def query_getsubset(self, args):
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
        searchfilter = args.pop('searchfilter', None)

        try:
            res = _getCubes(searchfilter, **args)
        except MarvinError as e:
            self.results['error'] = str(e)
            self.results['traceback'] = get_traceback(asstring=True)
        else:
            self.results['status'] = 1
            self.update_results(res)

        # this needs to be json.dumps until sas-vm at Utah updates to 2.7.11
        return Response(json.dumps(self.results), mimetype='application/json')

    @route('/getparamslist/', methods=['GET', 'POST'], endpoint='getparams')
    @av.check_args(use_params='query', required='paramdisplay')
    def getparamslist(self, args):
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
        paramdisplay = args.pop('paramdisplay', 'all')
        q = Query(mode='local', release=args['release'])
        if paramdisplay == 'all':
            params = q.get_available_params('all')
        elif paramdisplay == 'best':
            params = bestparams
        self.results['data'] = params
        self.results['status'] = 1
        output = jsonify(self.results)
        return output

    @route('/cleanup/', methods=['GET', 'POST'], endpoint='cleanupqueries')
    @av.check_args(use_params='query', required='task')
    def cleanup(self, args):
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
        task = args.pop('task', None)
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
