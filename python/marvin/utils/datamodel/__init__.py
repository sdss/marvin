# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 10:31:11
# @Last modified by:   andrews
# @Last modified time: 2017-10-02 14:10:71

from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from six import with_metaclass


class MetaDataModel(type):
    ''' MetaClass to construct a new DataModelList class '''
    def __new__(cls, name, parents, dict):
        if 'base' in dict:
            item = list(dict['base'].items())[0]
            dict['base_name'] = item[0].strip()
            dict['base_model'] = item[1]
        return super(MetaDataModel, cls).__new__(cls, name, parents, dict)


class DataModelList(with_metaclass(MetaDataModel, OrderedDict)):
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
