
import contextlib
import functools
import os
import re
from copy import deepcopy

from brain.utils.general import build_routemap
from flask import current_app
from flask_jwt_extended import fresh_jwt_required
from webargs import ValidationError, fields, validate
from webargs.flaskparser import parser, use_args, use_kwargs

from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.utils.datamodel.dap import datamodel as dm
from marvin.utils.general import validate_jwt


def plate_in_range(val):
    if int(val) < 6500:
        raise ValidationError('Plateid must be > 6500')


# List of global View arguments across all API routes
viewargs = {'name': fields.String(required=True, location='view_args', validate=[validate.Length(min=4),
                                  validate.Regexp('^[0-9-]*$')]),
            'galid': fields.String(required=True, location='view_args', validate=[validate.Length(min=4),
                                   validate.Regexp('^[0-9-]*$')]),
            'bintype': fields.String(required=True, location='view_args'),
            'template': fields.String(required=True, location='view_args'),
            'property_name': fields.String(required=True, location='view_args'),
            'channel': fields.String(required=True, location='view_args'),
            'binid': fields.Integer(required=True, location='view_args', validate=validate.Range(min=-1, max=5800)),
            'plateid': fields.String(required=True, location='view_args', validate=[validate.Length(min=4, max=5),
                                     plate_in_range]),
            'x': fields.Integer(required=True, location='view_args', validate=validate.Range(min=0, max=100)),
            'y': fields.Integer(required=True, location='view_args', validate=validate.Range(min=0, max=100)),
            'mangaid': fields.String(required=True, location='view_args', validate=validate.Length(min=4, max=20)),
            'paramdisplay': fields.String(required=True, location='view_args', validate=validate.OneOf(['all', 'best'])),
            'cube_extension': fields.String(required=True, location='view_args',
                                            validate=validate.OneOf(['flux', 'ivar', 'mask',
                                                                     'disp', 'predisp',
                                                                     'specres', 'specresd',
                                                                     'prespecres',
                                                                     'prespecresd'])),
            'modelcube_extension': fields.String(required=True, location='view_args',
                                                 validate=validate.OneOf(['flux', 'ivar', 'mask',
                                                                          'model', 'emline',
                                                                          'emline_base',
                                                                          'emline_mask'])),
            'colname': fields.String(required=True, location='view_args', allow_none=True),
            'fiberid': fields.Integer(required=True, location='view_args',
                                      validate=validate.Range(min=-1, max=5800)),
            }

# List of all form parameters that are needed in all the API routes
# allow_none = True allows for the parameter to be non-existent when required=False and missing is not set
# setting missing = None by itself also works except when the parameter is also required
# (i.e. required=True + missing=None does not trigger the "required" validation error when it should)
params = {'query': {'searchfilter': fields.String(allow_none=True),
                    'paramdisplay': fields.String(allow_none=True, validate=validate.OneOf(['all', 'best'])),
                    'task': fields.String(allow_none=True, validate=validate.OneOf(['clean', 'getprocs'])),
                    'start': fields.Integer(allow_none=True, validate=validate.Range(min=0)),
                    'end': fields.Integer(allow_none=True, validate=validate.Range(min=0)),
                    'offset': fields.Integer(allow_none=True, validate=validate.Range(min=0)),
                    'limit': fields.Integer(missing=100, validate=validate.Range(max=50000)),
                    'sort': fields.String(allow_none=True),
                    'order': fields.String(missing='asc', validate=validate.OneOf(['asc', 'desc'])),
                    'rettype': fields.String(allow_none=True, validate=validate.OneOf(['cube', 'spaxel', 'maps', 'rss', 'modelcube'])),
                    'returnparams': fields.DelimitedList(fields.String(), allow_none=True),
                    'defaults': fields.DelimitedList(fields.String(), allow_none=True),
                    'targets': fields.DelimitedList(fields.String(), allow_none=True),
                    'quality': fields.DelimitedList(fields.String(), allow_none=True),
                    'return_all': fields.Boolean(allow_none=True),
                    'format_type': fields.String(allow_none=True, validate=validate.OneOf(['list', 'listdict', 'dictlist'])),
                    'caching': fields.Boolean(allow_none=True),
                    'query_type': fields.String(allow_none=True, validate=validate.OneOf(['raw', 'core', 'orm']))
                    },
          'search': {'searchbox': fields.String(required=True),
                     'parambox': fields.DelimitedList(fields.String(), allow_none=True)
                     },
          'index': {'galid': fields.String(allow_none=True, validate=[validate.Length(min=4), validate.Regexp('^[0-9-]*$')]),
                    'mplselect': fields.String(allow_none=True, validate=validate.Regexp('^(MPL-|DR)([0-9]{1,2}$)'))
                    },
          'galaxy': {'plateifu': fields.String(allow_none=True, validate=validate.Length(min=8, max=11)),
                     'toggleon': fields.String(allow_none=True, validate=validate.OneOf(['true', 'false'])),
                     'image': fields.Url(allow_none=True),
                     'imheight': fields.Integer(allow_none=True, validate=validate.Range(min=0, max=1000)),
                     'imwidth': fields.Integer(allow_none=True, validate=validate.Range(min=0, max=1000)),
                     'type': fields.String(allow_none=True, validate=validate.OneOf(['optical', 'heatmap'])),
                     'x': fields.String(allow_none=True),
                     'y': fields.String(allow_none=True),
                     'mousecoords[]': fields.List(fields.String(), allow_none=True),
                     'bintemp': fields.String(allow_none=True),
                     'params[]': fields.List(fields.String(), allow_none=True)
                     }
          }


# Add a custom Flask session location handler
@parser.location_handler('session')
def parse_session(req, name, field):
    from flask import session as current_session
    value = current_session.get(name, None)
    return value


class ArgValidator(object):
    ''' Web/API Argument validator

    This class controls the parameter / argument validation for all API and Web route
    parameters, both in the views (url names), and in the request body (get/post parameters).
    This uses the Python package webargs and marshmallow to define and control all
    validation of parameters. See `webargs <https://webargs.readthedocs.io/en/latest/>`_ and
    `marshmallow <https://marshmallow.readthedocs.io/en/latest/>`_.

    There are two global dictionaries defining the argument validations.  `view_args` defines the
    list of all named arguments in the api/web Flask routes, e.g. `/marvin/api/cubes/<name>/` or
    `/marvin/api/maps/<name>/<bintype>/<template>/`.   `params` defines the list of all parameters
    used in routes as form arguments in GET or POST requests, in both the web and api Flask routes.
    Each entry is a dictionary of {page: dict of parameter validators}.  For example, the `query` key contains
    parameters related to sending requests to the API or Web Query views.

    You can use the validator in two ways, as a decorator, or manually inside a function.
    To use the validator as a decorator, decorate your route with the `check_args` method.  Your
    route function must contain an args input variable which is a dictionary of the validated
    parameters as checked by the decorator. See :meth:`marvin.api.ArgValidator.check_args` for more info.

    To use the validator manually, call the `manual_parse` method inside the function. See
    :meth:`marvin.api.ArgValidator.manual_parse` for more info.

    Example:
        >>> # use as a decorator (checks any view url arguments)
        >>> from marvin.api.base import arg_validate as av
        >>> @route('/myroute/')
        >>> @av.check_args()
        >>> def do_stuff(args):
        >>>     return stuff
        >>>
        >>> # use manually inside a Flask view function (checks the galaxy request parameters
        >>> # and required plateifu is set)
        >>> from flask import request
        >>> def do_stuff(self):
        >>>     args = av.manual_parse(self, request, use_params='galaxy', required='plateifu')
        >>>     return stuff

    '''

    def __init__(self, urlmap={}):
        self.release = None
        self.endpoint = None
        self.dapver = None
        self.urlmap = urlmap
        self.base_args = {'release': fields.String(required=True,
                          validate=validate.Regexp('^(MPL-|DR)([0-9]{1,2}$)'))}
        self.use_params = None
        self._required = None
        self._setmissing = None
        self._main_kwargs = {}
        self.final_args = {}
        self.final_args.update(self.base_args)

        self._parser = parser
        self.use_args = use_args
        self.use_kwargs = use_kwargs

    def _check_urlmap(self):
        ''' Check that urlmap exists and build it if not '''
        if not self.urlmap:
            urlmap = build_routemap(current_app)
            self.urlmap = urlmap

    def _reset_final_args(self):
        ''' Resets the final args dict '''
        self.final_args = {}
        self.final_args.update(self.base_args)
        # also reset the bintemp validators so toggles between MPL-4 and 5+ don't cause validation errors
        params['galaxy']['bintemp'].validators = []

    def _get_url(self):
        ''' Retrieve the URL route from the map based on the request endpoint '''
        self._check_urlmap()
        blue, end = self.endpoint.split('.', 1)
        url = self.urlmap[blue][end]['url']
        # if the blueprint is not api, add/remove session option location
        if blue == 'api':
            if 'session' in parser.locations:
                pl = list(parser.locations)
                pl.remove('session')
                parser.locations = tuple(pl)
        else:
            if 'session' not in parser.locations:
                parser.locations += ('session', )

        return url

    def _extract_view_args(self):
        ''' Extract any view argument parameters contained in the URL route '''
        url = self._get_url()
        url_viewargs = re.findall(r'{(.*?)}', url)
        return url_viewargs

    def _add_param_args(self):
        ''' Adds all appropriate form arguments into dictionary for validation '''

        # get the url
        url = self._get_url()

        # check list or not
        self.use_params = [self.use_params] if not isinstance(self.use_params, (list, tuple)) else self.use_params

        for local_param in self.use_params:
            if local_param in params:
                # update param validation
                if self._required or self._setmissing:
                    newparams = self.update_param_validation(local_param)
                else:
                    newparams = deepcopy(params)
                # add to params final args
                self.final_args.update(newparams[local_param])

    def _set_params_required(self, subset):
        ''' Set the param validation required parameter '''

        # make list or not
        self._required = [self._required] if not isinstance(self._required, (list, tuple)) else self._required

        # update the required attribute
        for req_param in self._required:
            if req_param in subset.keys():
                subset[req_param].required = True
                subset[req_param].allow_none = False

            if req_param == 'bintemp':
                bintemps = self._get_bin_temps()
                subset[req_param].validate = validate.OneOf(bintemps)
                subset[req_param].validators.append(validate.OneOf(bintemps))

        return subset

    def _set_params_missing(self, subset):
        ''' Set the param validation missing parameter '''

        for miss_field in subset.values():
            miss_field.missing = None

        return subset

    def update_param_validation(self, name):
        ''' Update the validation of form params '''

        # deep copy the global parameter dict
        newparams = deepcopy(params)
        subset = newparams[name]

        # Set required params
        if self._required:
            subset = self._set_params_required(subset)

        # Set missing params
        if self._setmissing:
            subset = self._set_params_missing(subset)

        # return the new params
        newparams[name] = subset
        return newparams

    def _add_view_args(self):
        ''' Adds all appropriate View arguments into dictionary for validation '''
        local_viewargs = self._extract_view_args()

        # check if any local_view args need new validation
        props = ['bintype', 'template', 'property_name', 'channel']
        ismatch = set(local_viewargs) & set(props)
        if ismatch:
            self.update_view_validation()

        # add only the local view args to the final arg list
        if local_viewargs:
            for varg in local_viewargs:
                self.final_args.update({varg: viewargs[varg]})

    def _update_viewarg(self, name, choices):
        ''' Updates the global View arguments validator '''
        viewargs[name] = fields.String(required=True, location='view_args', validate=validate.OneOf(choices))
        # viewargs[name].validate = validate.OneOf(choices)

    def _get_bin_temps(self):
        ''' Gets the bintemps for a given release '''
        bintemps = dm[self.dapver].get_bintemps()
        return bintemps

    def update_view_validation(self):
        ''' Update the validation of DAP MPL specific names based on the datamodel '''

        # update all the dap datamodel specific options
        bintemps = self._get_bin_temps()
        bintypes = list(set([b.split('-', 1)[0] for b in bintemps]))
        temps = list(set([b.split('-', 1)[1] for b in bintemps]))
        properties = [gr.name for gr in dm[self.dapver].properties.extensions]
        channels = list(set([i.channel.name for i in dm[self.dapver].properties
                             if i.channel is not None])) + ['None']

        # update the global viewargs for each property
        propfields = {'bintype': bintypes, 'template': temps, 'property_name': properties,
                      'channel': channels}
        for key, val in propfields.items():
            self._update_viewarg(key, val)

    def create_args(self):
        ''' Build the final argument list for webargs validation '''

        # get the dapver
        #drpver, self.dapver = config.lookUpVersions(self.release)

        try:
            drpver, self.dapver = config.lookUpVersions(self.release)
        except MarvinError as e:
            pass

        # reset the final args
        self._reset_final_args()

        # add view args to the list
        self._add_view_args()

        # add form param args to the list
        if self.use_params:
            self._add_param_args()

    def _pop_kwargs(self, **kwargs):
        ''' Pop all non webargs kwargs out of the main kwarg dict '''

        webargs_kwargs = ['req', 'locations', 'as_kwargs', 'validate']
        tempkwargs = kwargs.copy()
        for key in kwargs:
            if key not in webargs_kwargs:
                tmp = tempkwargs.pop(key, None)
        self._main_kwargs = tempkwargs

    def _check_mainkwargs(self, **kwargs):
        self.use_params = kwargs.pop('use_params', None)
        self._required = kwargs.pop('required', None)
        self._setmissing = kwargs.pop('set_missing', None)

    def _get_release_endpoint(self, view):
        ''' get the release and endpoint if you can '''
        self.release = view._release
        self.endpoint = view._endpoint

    def check_args(self, **mainkwargs):
        ''' Checks the input view and parameter arguments for validation using webargs

        This is a decorator and modifies the standard webargs.flaskparser.use_args decorator.
        check_args by default will check and validation all View parameter arguments.  It will
        optionally check Request parameter arguments when the `use_params` keyword is set.

        Parameters:
            use_params (str|list):
                one (or list) of the param keys defining which sets of parameters to consider
                for validation
            required (str|list):
                string or list of string parameter names that are required as input

        Returns:
            A dictionary of arguments if all passed validation or an error dictionary specifying
            any missing parameters, or parameters that failed validation.  There errors are already
            caught properly in the Marvin system.

        Example:
            >>> # to validate all input view args
            >>> @av.check_args()
            >>>
            >>> # to validate input query request parameters
            >>> @av.check_args(use_params='query', required='searchfilter')

        '''
        # decorator used to grab the release and endpoint of the route
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # args and kwargs here are the view function args and kwargs
                self._get_release_endpoint(args[0])
                # check the kwargs for any parameters
                self._check_mainkwargs(**mainkwargs)
                # create the arguments dictionary
                self.create_args()
                # pop all the kwargs
                self._pop_kwargs(**mainkwargs)
                # pass into webargs use_args (use_args is a decorator in itself)
                newfunc = self.use_args(self.final_args, **self._main_kwargs)(func)
                return newfunc(*args, **kwargs)
            return wrapper

        return decorator

    def check_kwargs(self, **kwargs):
        kwargs['as_kwargs'] = True
        return self.check_args(**kwargs)

    def check_release(self, **kwargs):
        ''' Checks only the release '''
        return self.use_kwargs(self.base_args)

    def list_params(self, param_type=None):
        ''' List the globally defined parameters for validation
        '''
        total = {'viewargs': viewargs, 'params': params}

        if param_type == 'viewargs':
            return total['viewargs']
        elif param_type == 'params':
            return total['params']
        else:
            return total

    def manual_parse(self, view, req, **mainkwargs):
        ''' Manually parse the args

        Manually parses and validates the input arguements of a Flask route

        Parameters:
            view:
                The instance of the Flask View
            req:
                The Flask request object
            use_params (str|list):
                one (or list) of the param keys defining which sets of parameters to consider
                for validation
            required (str|list):
                string or list of string parameter names that are required as input
            makemulti (bool):
                If True, converts the arguments dictionary into an ImmutableMultiDict object

        Returns:
            A dictionary of arguments if all passed validation or an error dictionary specifying
            any missing parameters, or parameters that failed validation.  There errors are already
            caught properly in the Marvin system.

        Example:
            >>> # to validate on the Galaxy View page, and check the galaxy request parameters
            >>> # and require that the plateifu parameter is set
            >>> args = av.manual_parse(self, request, use_params='galaxy', required='plateifu')

        '''
        self._get_release_endpoint(view)
        url = self._get_url()
        self._check_mainkwargs(**mainkwargs)
        self.create_args()
        self._pop_kwargs(**mainkwargs)
        newargs = parser.parse(self.final_args, req, force_all=True, **self._main_kwargs)

        # see if we make it a multidict
        makemulti = mainkwargs.get('makemulti', None)
        if makemulti:
            from werkzeug.datastructures import ImmutableMultiDict
            newargs = ImmutableMultiDict(newargs.copy())
        return newargs


# Creates a context manager for executing code with DB off but then turning it on again after that.

@contextlib.contextmanager
def db_off():
    config.forceDbOff()
    yield
    config.forceDbOn()


def set_api_decorators():
    ''' Sets the API decorators '''
    if os.environ.get('PUBLIC_SERVER', None):
        from brain.utils.general.decorators import public
        decorators = [public]
    else:
        decorators = [fresh_jwt_required, validate_jwt]

    return decorators
