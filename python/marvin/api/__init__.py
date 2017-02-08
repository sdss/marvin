
import re
import functools
from copy import deepcopy
from flask import request
from brain.api.base import processRequest
from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.utils.dap.datamodel import dap_datamodel as dm
from marvin.tools.maps import _get_bintemps
from webargs import fields, validate
from webargs.flaskparser import use_args, use_kwargs

config.use_sentry = False


def parse_params(request):
    """Parses the release from any type of request."""

    form = processRequest(request)
    print('parse', form)
    release = form.get('release', None)

    return release


# List of global View arguments across all API routes
viewargs = {'name': fields.String(required=True, location='view_args', validate=validate.Length(min=4)),
            'bintype': fields.String(required=True, location='view_args'),
            'template_kin': fields.String(required=True, location='view_args'),
            'property_name': fields.String(required=True, location='view_args'),
            'channel': fields.String(required=True, location='view_args'),
            'binid': fields.Integer(required=True, location='view_args', validate=validate.Range(min=-1, max=5800)),
            'plateid': fields.String(required=True, location='view_args', validate=validate.Length(min=4, max=5)),
            'x': fields.Integer(required=True, location='view_args', validate=validate.Range(min=0, max=100)),
            'y': fields.Integer(required=True, location='view_args', validate=validate.Range(min=0, max=100)),
            'mangaid': fields.String(required=True, location='view_args', validate=validate.Length(min=4, max=20))
            }

# List of all form parameters that are needed in all the API routes
params = {'query': {'searchfilter': fields.String(),
                    'paramdisplay': fields.String(validate=validate.OneOf(['all', 'best'])),
                    'task': fields.String(validate=validate.OneOf(['clean', 'getprocs'])),
                    'start': fields.Integer(validate=validate.Range(min=0)),
                    'end': fields.Integer(validate=validate.Range(min=0)),
                    'limit': fields.Integer(missing=100, validate=validate.Range(max=50000)),
                    'sort': fields.String(),
                    'order': fields.String(missing='asc', validate=validate.OneOf(['asc', 'desc'])),
                    'rettype': fields.String(validate=validate.OneOf(['cube', 'spaxel', 'maps', 'rss', 'modelcube'])),
                    'params': fields.DelimitedList(fields.String())
                    }
          }


class ArgValidator(object):
    ''' Web/API Argument validator '''

    def __init__(self, urlmap={}):
        self.release = None
        self.endpoint = None
        self.urlmap = urlmap
        self.base_args = {'release': fields.String(required=True, validate=validate.Regexp('MPL-[4-9]'))}
        self.use_params = None
        self._required = None
        self._main_kwargs = {}
        self.final_args = {}
        self.final_args.update(self.base_args)

    def _get_url(self):
        ''' Retrieve the URL route from the map based on the request endpoint '''
        blue, end = self.endpoint.split('.', 1)
        url = self.urlmap[blue][end]['url']
        print('blue, end', blue, end)
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
                if self._required:
                    newparams = self.update_param_validation(local_param)
                else:
                    newparams = deepcopy(params)

                # add to params final args
                self.final_args.update(newparams[local_param])

    def update_param_validation(self, name):
        ''' Update the validation of form params '''

        # make list or not
        self._required = [self._required] if not isinstance(self._required, (list, tuple)) else self._required

        # deep copy the global parameter dict
        newparams = deepcopy(params)
        subset = newparams[name]

        # update the required attribute
        for req_param in self._required:
            if req_param in subset.keys():
                subset[req_param].required = True

        # return the new params
        newparams[name] = subset
        return newparams

    def _add_view_args(self):
        ''' Adds all appropriate View arguments into dictionary for validation '''
        local_viewargs = self._extract_view_args()

        # check if any local_view args need new validation
        props = ['bintype', 'template_kin', 'property_name', 'channel']
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
        #viewargs[name].validate = validate.OneOf(choices)

    def update_view_validation(self):
        ''' Update the validation of DAP MPL specific names based on the datamodel '''

        # get the dapver
        drpver, dapver = config.lookUpVersions(self.release)

        # update all the dap datamodel specific options
        bintemps = _get_bintemps(dapver)
        bintypes = list(set([b.split('-', 1)[0] for b in bintemps]))
        temps = list(set([b.split('-', 1)[1] for b in bintemps]))
        properties = dm[dapver].list_names()
        channels = list(set(sum([i.channels for i in dm[dapver] if i.channels is not None], [])))

        # update the global viewargs for each property
        propfields = {'bintype': bintypes, 'template_kin': temps, 'property_name': properties,
                      'channel': channels}
        for key, val in propfields.items():
            self._update_viewarg(key, val)

    def create_args(self):
        ''' Build the final argument list for webargs validation '''

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

    def check_args(self, **mainkwargs):
        ''' Checks the input view and parameter arguments for validation using webargs

        This is a decorator and modifies the standard webargs.flaskparser use_args decorator

        '''

        # self note:  nothing can go out here since the decorator is called a lot of overwrites shit

        # decorator used to grab the release and endpoint of the route
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # args and kwargs here are the view function args and kwargs
                self.release = args[0]._release
                self.endpoint = args[0]._endpoint
                # check the kwargs for any parameters
                self._check_mainkwargs(**mainkwargs)
                # create the arguments dictionary
                self.create_args()
                # pop all the kwargs
                self._pop_kwargs(**mainkwargs)
                # pass into webargs use_args (use_args is a decorator in itself)
                newfunc = use_args(self.final_args, self._main_kwargs)(func)
                return newfunc(*args, **kwargs)
            return wrapper

        return decorator

    def check_kwargs(self, **kwargs):
        kwargs['as_kwargs'] = True
        return self.check_args(**kwargs)

    def check_release(self, **kwargs):
        ''' Checks only the release '''
        return use_kwargs(self.base_args)

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




