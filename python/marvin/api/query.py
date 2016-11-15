import json
from flask_classy import route
from flask import request
from marvin.tools.query import doQuery, Query
from marvin.core.exceptions import MarvinError
from marvin.api import parse_params
from marvin.api.base import BaseView


def _getCubes(searchfilter, params=None, rettype=None, start=None, end=None,
              limit=None, sort=None, order=None):
    """Run query locally at Utah."""

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
    """Class describing API calls related to queries.

    example query post:
    curl -X POST --data "searchfilter=nsa_redshift<0.1" http://sas.sdss.org/marvin2/api/query/cubes/
    """

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    @route('/cubes/', methods=['GET', 'POST'], endpoint='querycubes')
    def cube_query(self):
        ''' do a remote query '''
        searchfilter = self.results['inconfig'].get('searchfilter', None)
        params = self.results['inconfig'].get('params', None)
        rettype = self.results['inconfig'].get('returntype', None)
        limit = self.results['inconfig'].get('limit', 100)
        sort = self.results['inconfig'].get('sort', None)
        order = self.results['inconfig'].get('order', 'asc')
        print('inconfig', self.results['inconfig'])
        print('cube_query', searchfilter, params, limit)
        try:
            res = _getCubes(searchfilter, params=params, rettype=rettype,
                            limit=limit, sort=sort, order=order)
        except MarvinError as e:
            self.results['error'] = str(e)
        else:
            self.results['status'] = 1
            self.update_results(res)

        return json.dumps(self.results)

    @route('/cubes/getsubset', methods=['GET', 'POST'], endpoint='getsubset')
    def query_getsubset(self):
        ''' remotely grab a subset of values '''
        searchfilter = self.results['inconfig'].get('searchfilter', None)
        params = self.results['inconfig'].get('params', None)
        start = self.results['inconfig'].get('start', None)
        end = self.results['inconfig'].get('end', None)
        rettype = self.results['inconfig'].get('returntype', None)
        limit = self.results['inconfig'].get('limit', 100)
        sort = self.results['inconfig'].get('sort', None)
        order = self.results['inconfig'].get('order', 'asc')
        try:
            res = _getCubes(searchfilter, params=params, start=int(start),
                            end=int(end), rettype=rettype, limit=limit,
                            sort=sort, order=order)
        except MarvinError as e:
            self.results['error'] = str(e)
        else:
            self.results['status'] = 1
            self.update_results(res)

        return json.dumps(self.results)

    @route('/getparamslist/', methods=['GET', 'POST'], endpoint='getparams')
    def getparamslist(self):
        ''' Retrieve a list of all available input parameters into the query '''

        q = Query(mode='local')
        allparams = q.get_available_params()
        self.results['data'] = allparams
        self.results['status'] = 1
        output = json.dumps(self.results)
        return output

    @route('/cleanup/', methods=['GET', 'POST'], endpoint='cleanupqueries')
    def cleanup(self):
        ''' Clean up idle server-side queries or retrieve the list of them '''
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
        output = json.dumps(self.results)
        return output
