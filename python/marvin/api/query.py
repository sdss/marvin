import json
from flask.ext.classy import route
from flask import session as current_session
from marvin.api.base import BaseView
from marvin.tools.query import Query, doQuery
from marvin.tools.core import MarvinError


def _getCubes(strfilter):
    """Run query locally at Utah."""
    q, r = doQuery(strfilter)
    r.getAll()
    output = {'data': r.getListOf('plateifu'), 'query': str(r.query),
              'filter': strfilter}
    return output

# write tests for API: if it fails, then it returns some status with a -1 and
# an error message in the JSON

# fill in line 159 of tools/query


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    route_base = '/query/'

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    """example query post:
    curl -X POST --data "strfilter=nsa_redshift<0.1" http://cd057661.ngrok.io/api/query/cubes/
    """

    @route('/cubes/', methods=['GET', 'POST'], endpoint='cubes')
    def cube_query(self):
        strfilter = self.results['inconfig']['strfilter']
        try:
            res = _getCubes(strfilter)
        except MarvinError as e:
            self.results['error'] = e # NOT WORKING YET
        else:
            self.update_results(res)

        return json.dumps(self.results)

    @route('/webtable/', methods=['GET', 'POST'], endpoint='webtable')
    def webtable(self):
        ''' Do a query for the Bootstrap Table in Marvin web '''

        searchvalue = current_session['searchvalue']
        limit = self.results['inconfig'].get('limit', None)
        offset = self.results['inconfig'].get('offset', None)
        order = self.results['inconfig'].get('order', None)
        sort = self.results['inconfig'].get('sort', None)
        search = self.results['inconfig'].get('search', None)
        q, res = doQuery(searchvalue, limit=limit, order=order, sort=sort)
        print('stuff', sort, order, limit, offset, search)
        print(res.results[0])
        # sort
        #revorder = 'desc' in order
        #print('reverse', revorder, order)
        #res.results = sorted(res.results, key=lambda x: x.plateifu, reverse=revorder)
        # get subset on a given page
        results = res.getSubset(offset, limit=limit)
        rows = res.getDictOf('plateifu', format_type='listdict')
        output = {'total': res.count, 'rows': rows}
        output = json.dumps(output)
        return output
