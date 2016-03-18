import json
from flask import request
from flask.ext.classy import route
from marvin.api.base import BaseView


def _getCubes(query):
    """Run query locally at Utah."""
    # results = do_query(query)
    results = {'data': 'jackpot!'}
    return results


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    route_base = '/query/'
    # config.mode = 'remote'

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    """example query post:
    curl -i -H "Content-Type: application/json" -X POST -d '{"query":"nsa_redshift=0.01-0.015"}' http://localhost:5000/api/query/cubes/
    """

    @route('/cubes/', methods=['POST'])
    def cube_query(self):
        self.results['query'] = request.json['query']
        res = _getCubes(self.results['query'])
        self.update_results(res)
        return json.dumps(self.results)
