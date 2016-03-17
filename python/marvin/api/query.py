from flask.ext.classy import route
from marvin.api.base import BaseView
import json


class QueryView(BaseView):
    """Class describing API calls related to queries."""

    route_base = '/queries/'

    def index(self):
        self.results['data'] = 'this is a query!'
        return json.dumps(self.results)

    # @route('/plateifus/')
    # def get_plateifus(self, name):
    #     """This method performs a get request at the url route /queries/<id>"""
    #     cube, res = _getCube(name)
    #     self.update_results(res)
    #     if cube:
    #         self.results['data'] = {name: '{0},{1},{2},{3}'.format(
    #                                     name, cube.plate, cube.ra, cube.dec)}
    #     return json.dumps(self.results)

