import json
from flask import request
from flask.ext.classy import route
from marvin.api.base import BaseView
from marvin.tools.query import Query


def _getCubes(query):
    """Run query locally at Utah."""
    q = Query()
    q.set_filter(params=query)
    q.add_condition()
    r = q.run()
    output = {'data': ['-'.join((str(it.plate), it.ifu.name))
                       for it in r.results]}
    return output


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    route_base = '/query/'
    # config.mode = 'remote'

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    """example query post:
    curl -X POST --data "query=nsa_redshift<0.1" http://519f4f12.ngrok.io/api/query/cubes/
    """

    @route('/cubes/', methods=['GET', 'POST'])
    def cube_query(self):
        query = self.results['inconfig']['query']
        res = _getCubes(query)
        self.update_results(res)
        return json.dumps(self.results)
