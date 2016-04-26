import json
from flask.ext.classy import route
from flask import session as current_session
from brain.api.query import BrainQueryView
from marvin.tools.query import doQuery
from marvin.core import MarvinError


def _getCubes(searchfilter, params_out=['plateifu']):
    """Run query locally at Utah."""

    q, r = doQuery(searchfilter)
    r.getAll()
    output = dict(data=r.getListOf('plateifu'),
                  query=r.showQuery(), filter=searchfilter)
    # output = dict(data={p: r.getListOf(p) for p in params_out},
    #               query=r.showQuery(), filter=searchfilter)
    return output


class QueryView(BrainQueryView):
    """Class describing API calls related to queries."""

    """example query post:
    curl -X POST --data "searchfilter=nsa_redshift<0.1" http://cd057661.ngrok.io/api/query/cubes/

    curl -X POST --data "searchfilter=nsa_redshift<0.1&params_out=mangaid" http://localhost:5000/api/query/cubes/
    """

    @route('/cubes/', methods=['GET', 'POST'], endpoint='querycubes')
    def cube_query(self):
        searchfilter = self.results['inconfig']['searchfilter']
        # params_tmp = self.results['inconfig']['params_out']
        # params_out = params_tmp.split(',')
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
        print('webtable', searchvalue, self.results['inconfig'])
        limit = self.results['inconfig'].get('limit', None)
        offset = self.results['inconfig'].get('offset', None)
        order = self.results['inconfig'].get('order', None)
        sort = self.results['inconfig'].get('sort', None)
        search = self.results['inconfig'].get('search', None)
        # do query
        q, res = doQuery(searchfilter=searchvalue, limit=limit, order=order, sort=sort)
        # get subset on a given page
        results = res.getSubset(offset, limit=limit)
        # get keys
        cols = res.mapColumnsToParams()
        # create output
        rows = res.getDictOf(format_type='listdict')
        output = {'total': res.count, 'rows': rows, 'columns': cols}
        print('webtable output', output)
        output = json.dumps(output)
        return output
