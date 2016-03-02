from __future__ import print_function
from marvin.tools.core import MarvinToolsClass, MarvinError
from flask.ext.sqlalchemy import BaseQuery
from marvin import config, session, datadb
from marvin.tools.query.results import Results
from marvin.tools.query.forms import MarvinForm
from sqlalchemy import or_, and_, bindparam
from operator import le, ge, gt, lt, eq, ne
from collections import defaultdict
import re
from sqlalchemy.dialects import postgresql
from functools import wraps

__all__ = ['Query']
opdict = {'<=': le, '>=': ge, '>': gt, '<': lt, '!=': ne, '=': eq}
# opdict = {'le': le, 'ge': ge, 'gt': gt, 'lt': lt, 'ne': ne, 'eq': eq}


# decorator
def updateConfig(f):
    ''' Decorator that updates the query object with new config info '''
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.query:
            self.query = self.query.params({'drpver': config.drpver})
        return f(self, *args, **kwargs)
    return wrapper


class Query(object):
    ''' can this be subclassed from sqlalchemy query '''

    def __init__(self, *args, **kwargs):

        # super(Query, self).__init__(*args, **kwargs)
        self.query = None
        self.params = None
        self.session = session
        self.filter = None
        self.joins = []
        self.myforms = defaultdict(str)
        #self.drpver = config.drpver
        #self.dapver = config.dapver

    def set_params(self, params=None):
        """Set parameters."""

        """
        there should be something to easily allow users to add in parameters in Tool-Mode to query with.  for now, do make believe
        """

        if not params:
            params = {'nsa_redshift': 0.012}
        self.params = params
        print('query params', self.params)
        ''' example wtform for sample table only; this input ideally should be a multidict , then web/api+local versions can be same '''
        # self.sampform = SampleForm(**self.params)

        self.marvinform = MarvinForm()
        for key in self.params.keys():
            self.myforms[key] = self.marvinform.callInstance(self.marvinform._param_form_lookup[key], params=self.params)

    def add_condition(self):
        """ Add a condition based on input form data. """
        # if self.mode == 'remote':
        # elif self.mode == 'api':
        ''' Maybe need super MarvinForm class that can contain all parameters, mapped to their invididual forms???.  for now, example it with SampleForm only'''

        for form in self.myforms.values():
            if not self._tableInQuery(form.Meta.model.__tablename__):
                self.joins.append(form.Meta.model.__tablename__)
                self.query = self.query.join(form.Meta.model)

            # build the filter
            self.build_filter(form)

        # add the filter to the query
        if not isinstance(self.filter, type(None)):
            self.query = self.query.filter(self.filter)

    def build_filter(self, form):
        ''' build a set of filter conditions to load into sqlalchemy filter ; needs to be generalized '''

        for key, value in form.data.items():
            # Only do if a value is present
            if value:
                # check for comparative operator
                iscompare = any([s in value for s in opdict.keys()])

                if iscompare:
                    # do operator comparison

                    # separate operator and value
                    value.strip()
                    try:
                        ops, number = value.split()
                    except ValueError:
                        match = re.match(r"([<>=!]+)([0-9.]+)", value, re.I)
                        if match:
                            ops = match.groups()[0]
                            number = match.groups()[1]
                    op = opdict[ops]

                    # Make the filter
                    myfilter = op(form.Meta.model.__table__.columns.__getitem__(key), number)
                else:
                    # do range or equality comparison
                    vals = re.split('[-,]', value.strip())
                    if len(vals) == 1:
                        # do straight equality comparison
                        number = vals[0]
                        if 'IFU' in str(form.Meta.model):
                            # Make the filter
                            myfilter = form.Meta.model.__table__.columns.__getitem__(key).startswith('{0}'.format(number))
                        else:
                            # Make the filter
                            myfilter = form.Meta.model.__table__.columns.__getitem__(key) == number
                    else:
                        # do range comparison
                        low, up = vals
                        # Make the filter
                        myfilter = and_(form.Meta.model.__table__.columns.__getitem__(key) >= low, form.Meta.model.__table__.columns.__getitem__(key) <= up)

                # Add to filter
                if isinstance(self.filter, type(None)):
                    self.filter = and_(myfilter)
                else:
                    self.filter = and_(self.filter, myfilter)

    @updateConfig
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

    @updateConfig
    def show(self, prop=None):
        ''' Prints info '''
        assert prop in [None, 'query', 'tables', 'joins', 'filter'], 'Input must be query, joins, or filter'
        if not prop or 'query' in prop:
            print(self.query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
        elif prop == 'tables':
            print(self.joins)
        elif prop == 'filter':
            print(self.filter.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
        else:
            print(self.__getattribute__(prop))

    def reset(self):
        ''' Resets the query '''
        self.filter = None
        self.myforms = None
        self.query = None
        self.params = None
        self.joins = None

    @updateConfig
    def _createBaseQuery(self, param=None):
        ''' create the base query session object '''
        if not param:
            self.query = self.session.query(datadb.Cube).join(datadb.PipelineInfo, datadb.PipelineVersion).filter(datadb.PipelineVersion.version == bindparam('drpver')).\
                params({'drpver': config.drpver})
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



