from __future__ import print_function
from marvin.tools.core import MarvinToolsClass, MarvinError
from flask.ext.sqlalchemy import BaseQuery
from marvin import session, datadb
from marvin.tools.query.results import Results
from marvin.tools.query.forms import SampleForm
from sqlalchemy import or_, and_

__all__ = ['Query']


class Query(object):
    ''' can this be subclassed from sqlalchemy query '''

    def __init__(self, *args, **kwargs):

        # super(Query, self).__init__(*args, **kwargs)
        self.query = None
        self.params = None
        self.session = session
        # handle different modes

    def set_params(self, params=None):
        """Set parameters."""

        """
        there should be something to easily allow users to add in parameters in Tool-Mode to query with.  for now, do make believe
        """

        if not params:
            params = {'nsa_redshift': 0.012}
        self.params = params
        ''' example wtform for sample table only; this input ideally should be a multidict , then web/api+local versions can be same '''
        self.sampform = SampleForm(**self.params)

    def add_condition(self):
        """ Add a condition based on input form data. """
        # if self.mode == 'remote':
        # elif self.mode == 'api':
        ''' Maybe need super MarvinForm class that can contain all parameters, mapped to their invididual forms???.  for now, example it with SampleForm only'''

        f = self.build_filter()
        if not self._tableInQuery(self.samplform.Meta.model.__tablename__):
            self.query = self.query.join(self.sampform.Meta.model)

        self.query = self.query.filter(f)

    def build_filter(self):
        ''' build a set of filter conditions to load into sqlalchemy filter ; needs to be generalized '''
        f = None
        for key, val in self.sampform.data.items():
            if val:
                if not f:
                    f = and_(self.sampform.Meta.model.__table__.columns.__getitem__(key) < val)
                else:
                    f = and_(f, self.sampform.Meta.model.__table__.columns.__getitem__(key) < val)
        return f

    def run(self, qmode='all'):
        """ Run the query and return an instance of Marvin Results class to deal with results????

        does the switch happen here between local db (local or web) and remote API call? Query built locally first and ran here or there?
        Or does the entire query get built on server-side during API call, and only input form data is pushed to server - maybe this is better?

        """

        if qmode == 'all':
            res = self.query.all()
        elif qmode == 'one':
            res = self.query.one()
        elif qmode == 'first':
            res = self.query.first()
        elif qmode == 'count':
            res = self.query.count()
        return Results(results=res)

    def _createBaseQuery(self, param=None):
        ''' create the base query session object '''
        if not param:
            self.query = self.session.query(datadb.Cube)
        else:
            self.query = self.session.query(param)

    def _tableInQuery(self, name):
        ''' check if a given SQL table is already in the SQL query '''

        # check if a base query exists, this should probably be a decorator
        if not self.query:
            self._createBaseQuery()

        # do the check
        try:
            isin = name in str(self.query._from_obj[0])
        except IndexError as e:
            isin = False
        except AttributeError as e:
            if type(self.query) == str:
                isin = name in self.query
            else:
                isin = False
        return isin



