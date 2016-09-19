import json
from flask.ext.classy import route
from flask import session as current_session, request
from brain.api.query import BrainQueryView
from marvin.tools.query import doQuery, Query
from marvin.core import MarvinError
from marvin.api import parse_params


def _getCubes(searchfilter, params=None, start=None, end=None):
    """Run query locally at Utah."""

    drpver, dapver = parse_params(request)

    q, r = doQuery(searchfilter=searchfilter, returnparams=params, mode='local', drpver=drpver, dapver=dapver)
    results = r.results

    # get a subset
    if start:
        chunk = int(end)-int(start)
        results = r.getSubset(int(start), limit=chunk)

    output = dict(data=results, query=r.showQuery(),
                  filter=searchfilter, params=params,
                  queryparams_order=q.queryparams_order, count=len(results), totalcount=r.count)
    return output


class QueryView(BrainQueryView):
    """Class describing API calls related to queries.

    example query post:
    curl -X POST --data "searchfilter=nsa_redshift<0.1" http://sas.sdss.org/marvin2/api/query/cubes/
    """

    @route('/cubes/', methods=['GET', 'POST'], endpoint='querycubes')
    def cube_query(self):
        ''' do a remote query '''
        searchfilter = self.results['inconfig'].get('searchfilter', None)
        params = self.results['inconfig'].get('params', None)
        print('cube_query', searchfilter, params)
        try:
            res = _getCubes(searchfilter, params=params)
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
        try:
            res = _getCubes(searchfilter, params=params, start=int(start), end=int(end))
        except MarvinError as e:
            self.results['error'] = str(e)
        else:
            self.results['status'] = 1
            self.update_results(res)

        return json.dumps(self.results)

    @route('/getparamslist/', methods=['GET', 'POST'], endpoint='getparams')
    def getparamslist(self):
        ''' Retrieve a list of all available input parameters into the query '''

        q = Query()
        allparams = q.get_available_params()
        self.results['data'] = allparams
        self.results['status'] = 1
        output = json.dumps(self.results)
        return output
