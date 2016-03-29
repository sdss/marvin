import json
from flask.ext.classy import route
from flask import session as current_session
from marvin.api.base import BaseView
from marvin.tools.query import Query, doQuery


def _getCubes(query):
    """Run query locally at Utah."""
    q, r = doQuery(query)
    res = r.getAll()
    output = {'data': r.getListOf('plateifu')}
    return output

# expose aspects of query object or results object
# e.g., SQL query string (r.query)
# q.strfilter  # natural language


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    route_base = '/query/'

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    """example query post:
    curl -X POST --data "query=nsa_redshift<0.1" http://519f4f12.ngrok.io/api/query/cubes/
    """

    @route('/cubes/', methods=['GET', 'POST'], endpoint='cubes')
    def cube_query(self):
        query = self.results['inconfig']['query']
        res = _getCubes(query)
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
