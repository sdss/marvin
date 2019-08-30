# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-09-03 18:56:38
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-20 12:06:47

from __future__ import absolute_import, division, print_function

from brain.utils.general import compress_data
from brain.utils.general.decorators import public
from flask import Response, jsonify, redirect, stream_with_context, url_for
from flask_classful import route
from marvin import config
from marvin.api.base import arg_validate as av
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.tools.query import Query, doQuery
from marvin.utils.datamodel.query.base import bestparams
from marvin.utils.db import get_traceback
from marvin.web.extensions import limiter, celery


@celery.task
def test_sleep(t):
    import time
    print('celery2', celery)
    time.sleep(t)
    return {'result': 'I have slept', 'status': 'Task Completed', 'current': 10}


def _recombine_args(args):
    ''' Recombine any list keyword args intro strings '''
    newargs = {k: ','.join(v) if isinstance(v, list) else v for k, v in args.items()}
    return newargs


def _run_query(searchfilter, **kwargs):
    ''' Run the query and return the query and results '''

    release = kwargs.pop('release', None)
    kwargs['return_params'] = kwargs.pop('returnparams', None)
    kwargs['default_params'] = kwargs.pop('defaults', None)
    kwargs['return_type'] = kwargs.pop('rettype', None)

    try:
        q, r = doQuery(search_filter=searchfilter, release=release, **kwargs)
    except Exception as e:
        raise MarvinError('Query failed with {0}: {1}'.format(e.__class__.__name__, e))
    else:
        return q, r


def _get_runtime(query):
    ''' Retrive a dictionary of the runtime to pass back in JSON '''
    runtime = {'days': query._run_time.days, 'seconds': query._run_time.seconds, 'microseconds': query._run_time.microseconds}
    return runtime


def _getCubes(searchfilter, **kwargs):
    """Run query locally at Utah and format the output into the full JSON """

    # run the query
    q, r = _run_query(searchfilter, **kwargs)

    # get the subset keywords
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    limit = kwargs.get('limit', None)
    returnparams = kwargs.get('returnparams', None)

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
                  filter=searchfilter, params=q.params, returnparams=returnparams, runtime=_get_runtime(q),
                  queryparams_order=q._query_params_order, count=len(results), totalcount=r.totalcount)
    return output


def gen(query, compression=config.compression, params=None):
    ''' Generator for query results

    Parameters:
        query (obj):
            The SQLalchemy query object
        compresssion (str):
            The type of compression to use, e.g. 'json' or 'msgpack'

    Yields:
        A compressed result row of data to stream to the client

    '''
    for i, row in enumerate(query):
        if i == 0 and params:
            yield compress_data(params, compress_with=compression) + ';\n'
        yield compress_data(row, compress_with=compression) + ';\n'


def _get_column(results, colname, format_type=None):
    ''' Gets a column from a Query

    Parameters:
        results (obj):
            A set of Marvin Results
        colname (str):
            The name of the column to extract
        format_type (str):
            The format of the dictionary

    Returns:
        A list of data for that column
    '''

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


def _compressed_response(compression, results):
    ''' Compress the data before sending it back in the Response

    Parameters:
        compression (str):
            The compression type.  Either `json` or `msgpack`.
        results (dict):
            The current response dictionary

    Returns:
        A Flask Response object with the compressed data
    '''

    # pack the data
    mimetype = 'json' if compression == 'json' else 'octet-stream'
    try:
        packed = compress_data(results, compress_with=compression)
    except Exception as e:
        results['error'] = str(e)
        results['traceback'] = get_traceback(asstring=True)

    return Response(stream_with_context(packed), mimetype='application/{0}'.format(mimetype))


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

           GET /marvin/api/query/ HTTP/1.1
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

    @route('/stream/', methods=['GET', 'POST'], endpoint='stream')
    @av.check_args(use_params='query', required='searchfilter')
    def stream(self, args):
        ''' test query generator stream '''

        searchfilter = args.pop('searchfilter', None)
        compression = args.pop('compression', config.compression)
        mimetype = 'json' if compression == 'json' else 'octet-stream'

        release = args.pop('release', None)
        args['return_params'] = args.pop('returnparams', None)
        args['return_type'] = args.pop('rettype', None)
        q = Query(search_filter=searchfilter, release=release, **args)

        output = dict(data=None, chunk=q.limit, query=q.show(),
                      filter=searchfilter, params=q.params, returnparams=q.return_params, runtime=None,
                      queryparams_order=q._query_params_order, count=None, totalcount=None)

        return Response(stream_with_context(gen(q.query, compression=compression, params=q.params)), mimetype='application/{0}'.format(mimetype))

    @route('/test/', methods=['GET', 'POST'], endpoint='querytest')
    @av.check_args(use_params='query')
    def test(self, args):
        t = args.pop('start', 10)
        print('celery1', celery)
        task = test_sleep.delay(t)
        self.results['status'] = 1
        self.results['data'] = ['this is a test']
        #return jsonify(self.results)
        return jsonify(self.results)#, 202, {'Location': url_for('test.taskstatus', task_id=task.id)}

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

           GET /marvin/api/query/cubes/ HTTP/1.1
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
        # if return_all is True, perform a redirect to stream
        return_all = args.get('return_all', None)
        if return_all:
            args = _recombine_args(args)
            return redirect(url_for('api.stream', **args))

        searchfilter = args.pop('searchfilter', None)

        try:
            res = _getCubes(searchfilter, **args)
        except MarvinError as e:
            self.results['error'] = str(e)
            self.results['traceback'] = get_traceback(asstring=True)
        else:
            self.results['status'] = 1
            self.update_results(res)

        compression = args.pop('compression', config.compression)
        return _compressed_response(compression, self.results)

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

           GET /marvin/api/query/cubes/columns/plateifu/ HTTP/1.1
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
        colname = args.pop('colname', None)

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

        compression = args.pop('compression', config.compression)
        return _compressed_response(compression, self.results)
        #return Response(json.dumps(self.results), mimetype='application/json')

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

           GET /marvin/api/query/cubes/getsubset/ HTTP/1.1
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

        compression = args.pop('compression', config.compression)
        return _compressed_response(compression, self.results)
        # this needs to be json.dumps until sas-vm at Utah updates to 2.7.11
        #return Response(json.dumps(self.results), mimetype='application/json')

    @public
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

           GET /marvin/api/query/getparamslist/ HTTP/1.1
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
        if paramdisplay == 'all':
            try:
                params = Query.get_available_params('all', release=args['release'])
            except MarvinError as e:
                self.results['error'] = str(e)
                self.results['traceback'] = get_traceback(asstring=True)
                self.results['status'] = -1
                return jsonify(self.results)
        elif paramdisplay == 'best':
            params = bestparams
        self.results['data'] = params
        self.results['status'] = 1
        output = jsonify(self.results)
        return output

    @public
    @route('/getallparams/', methods=['GET', 'POST'], endpoint='getallparams')
    def getall(self):
        ''' Retrieve all the query parameters for all releases at once '''
        params = {}
        releases = config._allowed_releases.keys()
        for release in releases:
            params[release] = Query.get_available_params('all', release=release)
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

           GET /marvin/api/query/cleanup/ HTTP/1.1
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


    # @route('/status/<task_id>')
    # def taskstatus(self, task_id):
    #     task = add_test.AsyncResult(task_id)
    #     if task.state == 'PENDING':
    #         print('pending task', task.state)
    #         # job did not start yet
    #         response = {
    #             'state': task.state,
    #             'current': 0,
    #             'total': 1,
    #             'status': 'Pending...'
    #         }
    #     elif task.state != 'FAILURE':
    #         print('success task', task.state, task.info)
    #         response = {
    #             'state': task.state,
    #             'current': task.info.get('current', 0),
    #             'status': task.info.get('status', '')
    #         }
    #         if 'result' in task.info:
    #             response['result'] = task.info['result']
    #     else:
    #         print('other task', task.state)
    #         # something went wrong in the background job
    #         response = {
    #             'state': task.state,
    #             'current': 1,
    #             'status': str(task.info),  # this is the exception raised
    #         }
    #     return jsonify(response)
