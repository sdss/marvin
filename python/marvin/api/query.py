import json
from flask.ext.classy import route
from flask import session as current_session
from marvin.api.base import BaseView
from marvin.tools.query import doQuery
from marvin.tools.core import MarvinError


def _getCubes(searchfilter):
    """Run query locally at Utah."""
    q, r = doQuery(searchfilter)
    r.getAll()
    output = {'data': r.getListOf('plateifu'), 'query': r.showQuery(), 'filter': searchfilter}
    return output


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    route_base = '/query/'

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    """example query post:
    curl -X POST --data "strfilter=nsa_redshift<0.1" http://cd057661.ngrok.io/api/query/cubes/
    """

    @route('/cubes/', methods=['GET', 'POST'], endpoint='querycubes')
    def cube_query(self):
        searchfilter = self.results['inconfig']['searchfilter']
        try:
            res = _getCubes(searchfilter)
        except MarvinError as e:
            self.results['error'] = str(e)
        else:
            self.update_results(res)

        return json.dumps(self.results)

    @route('/webtable/', methods=['POST'], endpoint='webtable')
    def webtable(self):
        ''' Do a query for Bootstrap Table interaction in Marvin web '''

        # set parameters
        searchvalue = current_session['searchvalue']
        limit = self.results['inconfig'].get('limit', None)
        offset = self.results['inconfig'].get('offset', None)
        order = self.results['inconfig'].get('order', None)
        sort = self.results['inconfig'].get('sort', None)
        search = self.results['inconfig'].get('search', None)
        # do query
        q, res = doQuery(searchvalue, limit=limit, order=order, sort=sort)
        # get subset on a given page
        results = res.getSubset(offset, limit=limit)
        # create output
        rows = res.getDictOf('plateifu', format_type='listdict')
        output = {'total': res.count, 'rows': rows}
        output = json.dumps(output)
        return output
