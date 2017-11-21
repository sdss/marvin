# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 10:31:11
# @Last modified by:   Brian Cherinka
# @Last modified time: 2017-10-02 14:10:71

from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from marvin import config, log
import six
import os


class MetaDataModel(type):
    ''' MetaClass to construct a new DataModelList class '''
    def __new__(cls, name, parents, dict):
        if 'base' in dict:
            item = list(dict['base'].items())[0]
            dict['base_name'] = item[0].strip()
            dict['base_model'] = item[1]
        return super(MetaDataModel, cls).__new__(cls, name, parents, dict)


class DataModelList(six.with_metaclass(MetaDataModel, OrderedDict)):
    ''' Base Class for a list of DataModels '''

    def __init__(self, models=None):

        if models is not None:
            assert all([isinstance(model, self.base_model) for model in models]), \
                'values must be {0} instances.'.format(self.base_name)
            OrderedDict.__init__(self, ((model.release, model) for model in models))
        else:
            OrderedDict.__init__(self, {})

    def __setitem__(self, key, value):
        """Sets a new datamodel."""

        assert isinstance(value, self.base_model), 'value must be a {0}'.format(self.base_name)

        super(DataModelList, self).__setitem__(key, value)

    def __getitem__(self, release):
        """Returns model based on release and aliases."""

        if release in self.keys():
            return super(DataModelList, self).__getitem__(release)

        for model in self.values():
            if release in model.aliases:
                return model

        raise KeyError('cannot find release or alias {0!r}'.format(release))

    def __repr__(self):

        return repr([xx for xx in self.values()])

    def add_datamodel(self, dm):
        """Adds a new datamodel. Uses its release as key."""

        assert isinstance(dm, self.base_model), 'value must be a {0}'.format(self.base_name)

        self[dm.release] = dm


class DataModelLookup(object):
    ''' Class for lookups into the Marvin DataModel '''

    def __init__(self, release=None):
        from marvin.utils.datamodel.dap import datamodel as dap_dms
        from marvin.utils.datamodel.drp import datamodel as drp_dms
        from marvin.utils.datamodel.query import datamodel as query_dms

        self.release = release if release else config.release
        assert release in query_dms.keys(), 'release must be a valid MPL'

        # set datamodels for a given release
        self.dapdm = dap_dms[release]
        self.drpdm = drp_dms[release]
        self.querydm = query_dms[release]
        self._dm = None
        self.model_map = ['drp', 'dap', 'query']

    def __repr__(self):

        return ('<DataModelLookup release={0!r}>'.format(self.release))

    def check_value(self, value, model=None):
        ''' Check that a value is in the Marvin datamodel

        Searches a specified datamodel for a value.  If no model is specified,
        attempts to search all the datamodels.

        Parameters:
            value (str):
                The value to look up in the datamodels
            model (str):
                Optional name of the datamodel to search on. Can be drp, dap, or query.

        Returns:
            True if value found within a specified model.  When no model is specified,
            returns the name of the model the value was found in.
        '''

        assert isinstance(value, six.string_types), 'value must be a string'
        assert model in self.model_map + [None], 'model must be drp, dap, or query'

        indrp = value in self.drpdm
        indap = value in self.dapdm
        inquery = value in self.querydm
        true_map = [indrp, indap, inquery]

        if model == 'dap':
            self._dm = self.dapdm
            return indap
        elif model == 'drp':
            self._dm = self.drpdm
            return indrp
        elif model == 'query':
            self._dm = self.querydm
            return inquery
        else:
            # check all of them
            tootrue = sum([indrp, indap]) > 1
            if tootrue:
                subset = [i for i in model_map if true_map[self.model_map.index(i)]]
                raise ValueError('{0} found in multiple datamodels {1}. '
                                 'Fine-tune your value or try a specific model'.format(value, subset))
            else:
                model = 'drp' if indrp else 'dap' if indap else 'query' if inquery else None

            if not model:
                print('{0} not found in any datamodels.  Refine your syntax or check the MaNGA TRM!'.format(value))
            return model

    def get_value(self, value, model=None):
        ''' Get the property or parameter for a given value

        Parameters:
            value (str):
                The name of the value to retrieve
            model (str):
                The name of the model to get the value from

        Returns:
            The parameter or property from the datamodel

        '''

        # check the value
        checked = self.check_value(value, model=model)

        if checked is True:
            param = value == self._dm
        elif checked == 'drp':
            param = value == self.drpdm
        elif checked == 'dap':
            param = value == self.dapdm
        elif checked == 'query':
            param = value == self.querydm
        elif not checked:
            print('No matching parameter found for {0} in model {1}'.format(value, model))
            param = None

        return param

    def write_csv(self, path=None, filename=None, model=None, overwrite=None, **kwargs):
        ''' Writes the datamodels out to CSV '''

        assert model in self.model_map + [None], 'model must be drp, dap, or query'

        if model == 'query':
            self.querydm.write_csv(path=path, filename=filename, overwrite=overwrite, db=True)
        elif model == 'dap':
            self.dapdm.properties.write_csv(path=path, filename=filename, overwrite=overwrite, **kwargs)
            self.dapdm.models.write_csv(path=path, filename=filename, overwrite=overwrite, **kwargs)
        elif model == 'drp':
            self.drpdm.spectra.write_csv(path=path, filename=filename, overwrite=overwrite, **kwargs)
            self.drpdm.datacubes.write_csv(path=path, filename=filename, overwrite=overwrite, **kwargs)

    def write_csvs(self, overwrite=None, **kwargs):
        ''' Write out all models to CSV files '''

        for model in self.model_map:
            self.write_csv(model=model, overwrite=overwrite, **kwargs)






