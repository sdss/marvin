#!/usr/bin/env python3
# encoding: utf-8
#
# datamodel.py
#
# Created by José Sánchez-Gallego on 18 Sep 2016.
# Refactored by José Sánchez-Gallego on 6 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy as copy_mod
import itertools
import six

from collections import OrderedDict

import astropy.table as table
from astropy import units as u

from fuzzywuzzy import fuzz as fuzz_fuzz
from fuzzywuzzy import process as fuzz_proc

from marvin.core.exceptions import MarvinError, MarvinNotImplemented


__ALL__ = ('DAPDataModelList', 'DAPDataModel', 'Bintype', 'Template', 'Property',
           'MultiChannelProperty', 'spaxel', 'datamodel')


spaxel = u.Unit('spaxel', represents=u.pixel, doc='A spectral pixel', parse_strict='silent')


def get_best_fuzzy(value, choices, min_score=30, scorer=fuzz_fuzz.WRatio, return_score=False):
    """Returns the best match in a list of choices using fuzzywuzzy."""

    bests = fuzz_proc.extractBests(value, choices, scorer=scorer, score_cutoff=min_score)

    if len(bests) == 0:
        best = None
    elif len(bests) == 1:
        best = bests[0]
    else:
        if bests[0][1] == bests[1][1]:
            best = None
        else:
            best = bests[0]

    if best is None:
        raise ValueError('cannot find a good match for {0!r}. '
                         'Your input value is too ambiguous.'.format(value))

    return best if return_score else best[0]


class DAPDataModelList(OrderedDict):
    """A dictionary of DAP datamodels."""

    def __init__(self, models=None):

        if models is not None:
            assert all([isinstance(model, DAPDataModel) for model in models]), \
                'values must be DAPDataModel instances.'
            OrderedDict.__init__(self, ((model.release, model) for model in models))
        else:
            OrderedDict.__init__(self, {})

    def __setitem__(self, key, value):
        """Sets a new datamodel."""

        assert isinstance(value, DAPDataModel), 'value must be a DAPDataModel'

        super(DAPDataModelList, self).__setitem__(key, value)

    def __getitem__(self, release):
        """Returns model based on release and aliases."""

        if release in self.keys():
            return super(DAPDataModelList, self).__getitem__(release)

        for model in self.values():
            if release in model.aliases:
                return model

        raise KeyError('cannot find release or alias {0!r}'.format(release))

    def __repr__(self):

        return repr([xx for xx in self.values()])

    def add_datamodel(self, dm):
        """Adds a new datamodel. Uses its release as key."""

        assert isinstance(dm, DAPDataModel), 'value must be a DAPDataModel'

        self[dm.release] = dm


class DAPDataModel(object):
    """A class representing a DAP datamodel, with bintypes, templates, properties, etc."""

    def __init__(self, release, bintypes=[], templates=[], properties=[], default_template=None,
                 default_bintype=None, aliases=[]):

        self.release = release
        self.bintypes = bintypes
        self.templates = templates

        self.aliases = aliases

        self._properties = []
        self.add_properties(properties, copy=True)

        self._default_bintype = None
        self.default_bintype = default_bintype

        self._default_template = None
        self.default_template = default_template

        assert len([bintype for bintype in self.bintypes if bintype.binned is False]) <= 1, \
            'a DAP datamodel can have only one unbinned bintype'

    @property
    def properties(self):
        """Returns the properties for this datamodel. MultiChannelProperties are unpacked."""

        full_list = []

        for prop in self._properties:
            if isinstance(prop, Property):
                full_list.append(prop)
            elif isinstance(prop, MultiChannelProperty):
                for multi_prop in prop:
                    full_list.append(multi_prop)
            else:
                raise ValueError('incorrect type of property {0!r}'.format(prop))

        return full_list

    @properties.setter
    def properties(self, value):
        """Raises an error if trying to set properties directly."""

        raise MarvinError('cannot set properties directly. Use add_properties() instead.')

    @property
    def groups(self):
        """Returns a list of properties. MultiChannelProperties are not unpacked."""

        return self._properties

    def __repr__(self):

        return ('<DAPDataModel release={0!r}, n_bintypes={1}, n_templates={2}, n_properties={3}>'
                .format(self.release, len(self.bintypes), len(self.templates),
                        len(self.properties)))

    def expand(self, **kwargs):
        """Expands attributes."""

        for kk in kwargs:
            if kk == 'properties':
                self.add_properties(kwargs[kk])
            else:
                setattr(kk, getattr(self, kk) + kwargs[kk])

    def add_property(self, prop, copy=True):
        """Adds a propety and sets its parent to this instance.

        If ``copy=True``, copies the property before adding it.

        """

        new_prop = copy_mod.copy(prop) if copy else prop
        new_prop.set_parent(self)

        self._properties.append(new_prop)

    def add_properties(self, properties, copy=True):
        """Adds a list of properties and sets their parent to this instance.

        If ``copy=True``, copies the properties before adding them.

        """

        for prop in properties:
            self.add_property(prop, copy=copy)

    def copy(self):
        """Returns a copy of the datamodel."""

        return copy_mod.deepcopy(self)

    def list_property_names(self):
        """Returns a list of all property names+channel."""

        return [prop.full() for prop in self.properties]

    def __getitem__(self, value):
        """Uses fuzzywuzzy to return the closest property match."""

        prop_names = [prop.name for prop in self.groups]

        # If the values matches exactly one of the property names, we return the property object.
        if value in prop_names:
            return self.groups[prop_names.index(value)]

        # Finds the best property+channel
        propety_channel_names = self.list_property_names()

        best = get_best_fuzzy(value, propety_channel_names)

        return self.properties[propety_channel_names.index(best)]

    @property
    def default_bintype(self):
        """Returns the default bintype."""

        return self._default_bintype

    @default_bintype.setter
    def default_bintype(self, value):
        """Sets the default bintype."""

        if isinstance(value, six.string_types):
            for bintype in self.bintypes:
                if value == bintype.name:
                    value = bintype

        if value not in self.bintypes and value is not None:
            raise ValueError('{0!r} not found in list of bintypes.'.format(value))

        self._default_bintype = value

    @property
    def default_template(self):
        """Returns the default template."""

        return self._default_template

    @default_template.setter
    def default_template(self, value):
        """Sets the default template."""

        if isinstance(value, six.string_types):
            for template in self.templates:
                if value == template.name:
                    value = template

        if value not in self.templates and value is not None:
            raise ValueError('{0!r} not found in list of templates.'.format(value))

        self._default_template = value

    def get_unbinned(self):
        """Returns the unbinned bintype."""

        for bintype in self.bintypes:
            if bintype.binned is False:
                return bintype

        raise MarvinError('cannot find unbinned bintype for '
                          'DAP datamodel for release {0}'.format(self.release))

    def get_bintype(self, value=None):
        """Returns a bintype whose name matches ``value``.

        Returns the default one if ``value=None``.

        """

        if value is None:
            return self.default_bintype

        if isinstance(value, Bintype):
            value = value.name

        for bintype in self.bintypes:
            if bintype.name == value:
                return bintype

        raise MarvinError('invalid bintype {0!r}'.format(value))

    def get_template(self, value=None):
        """Returns a template whose name matches ``value``.

        Returns the default one if ``value=None``.

        """

        if value is None:
            return self.default_template

        if isinstance(value, Template):
            value = value.name

        for template in self.templates:
            if template.name == value:
                return template

        raise MarvinError('invalid template {0!r}'.format(value))

    def get_bintemps(self, default=False):
        """Returns a list of all combinations of bintype and template."""

        if default:
            return '{0}-{1}'.format(self.default_bintype.name, self.default_template.name)

        bins = [bintype.name for bintype in self.bintypes]
        temps = [template.name for template in self.templates]

        return ['-'.join(item) for item in list(itertools.product(bins, temps))]

    def to_table(self, compact=True, pprint=False, description=False):
        """Returns an astropy table with all the properties in this model."""

        if compact:
            model_table = table.Table(
                None, names=['name', 'channels', 'ivar', 'mask', 'unit', 'description'],
                dtype=['S20', 'S300', bool, bool, 'S20', 'S500'])
        else:
            model_table = table.Table(
                None, names=['name', 'channel', 'ivar', 'mask', 'unit', 'description'],
                dtype=['S20', 'S20', bool, bool, 'S20', 'S500'])

        model_table.meta['release'] = self.release
        model_table.meta['bintypes'] = self.bintypes
        model_table.meta['templates'] = self.templates
        model_table.meta['default_bintype'] = self.default_bintype
        model_table.meta['default_template'] = self.default_template

        if compact:
            iterable = self.groups
        else:
            iterable = self.properties

        for prop in iterable:
            if isinstance(prop, MultiChannelProperty):
                channel = ', '.join(prop.channels)
                units = [pp.unit.to_string() for pp in prop]
                if len(set(units)) == 1:
                    unit = units[0]
                else:
                    unit = 'multiple'
            else:
                channel = '' if not prop.channel else prop.channel
                unit = prop.unit.to_string()

            model_table.add_row((prop.name, channel, prop.ivar, prop.mask, unit, prop.description))

        if not description:
            model_table.remove_column('description')

        if pprint:
            model_table.pprint()

        return model_table


class Bintype(object):

    def __init__(self, bintype, binned=True, default=False, n=None):

        self.name = bintype
        self.binned = binned
        self.n = n

    def __repr__(self):

        return '<Bintype {0!r}, binned={1!r}>'.format(self.name, self.binned)


class Template(object):

    def __init__(self, template, n=None):

        self.name = template
        self.n = n

    def __repr__(self):

        return '<Template {0!r}>'.format(self.name)


class Property(object):

    def __init__(self, name, channel=None, ivar=False, mask=False, unit=None, scale=1,
                 parent=None, binid=None, description=''):

        self.name = name
        self.channel = channel

        self.ivar = ivar
        self.mask = mask
        self.binid = binid

        self.scale = scale if scale is not None else 1
        self.unit = unit if unit is not None else u.dimensionless_unscaled

        self.description = description

        self.parent = parent

    def set_parent(self, parent):
        """Sets parent."""

        assert isinstance(parent, DAPDataModel), 'parent must be a DAPDataModel'

        self.parent = parent

    def full(self):
        """Returns the name + channel string."""

        if self.channel:
            return self.name + '_' + self.channel

        return self.name

    def get_binid(self):
        """Returns the binid property associated with this property."""

        if self.name == 'binid':
            raise MarvinError('binid has not associated binid (?!)')

        if self.parent is None:
            raise MarvinError('a parent needs to be defined to get an associated binid.')

        if self.binid is None:
            return self.parent['binid']
        else:
            raise MarvinNotImplemented('hybrid binning has not yet been implemented.')

    def db_column(self, ext=None):
        """Returns the name of the DB column containing this property."""

        assert ext is None or ext in ['ivar', 'mask'], 'invalid extension'

        if ext is None:
            return self.full()

        if ext == 'ivar':
            assert self.ivar is True, 'no ivar for property {0!r}'.format(self.full())
            return self.name + '_ivar' + ('_{0}'.format(self.channel) if self.channel else '')

        if ext == 'mask':
            assert self.mask is True, 'no mask for property {0!r}'.format(self.full())
            return self.name + '_mask' + ('_{0}'.format(self.channel) if self.channel else '')

    def __repr__(self):

        return '<Property {0!r}, release={1!r}, channel={2!r}, unit={3!r}>'.format(
            self.name, self.parent.release if self.parent else None,
            self.channel, self.unit.to_string())


class MultiChannelProperty(list):

    def __init__(self, name, channels=[], units=None, scales=None, **kwargs):

        self.name = name

        self.ivar = kwargs.get('ivar', False)
        self.mask = kwargs.get('mask', False)
        self.description = kwargs.get('description', '')

        self.parent = kwargs.get('parent', None)

        property_list = []
        for ii in range(len(channels)):
            channel = channels[ii]
            unit = units if units is None or not isinstance(units, (list, tuple)) else units[ii]
            scale = scales \
                if scales is None or not isinstance(scales, (list, tuple)) else scales[ii]

            property_list.append(Property(name, channel=channel, unit=unit, scale=scale, **kwargs))

        list.__init__(self, property_list)

    @property
    def channels(self):
        return [prop.channel for prop in self]

    def set_parent(self, parent):
        """Sets parent for the instance and all listed Property objects."""

        assert isinstance(parent, DAPDataModel), 'parent must be a DAPDataModel'

        self.parent = parent

        for prop in self:
            prop.parent = parent

    def __getitem__(self, value):
        """Uses fuzzywuzzy to get a channel."""

        if not isinstance(value, six.string_types):
            return super(MultiChannelProperty, self).__getitem__(value)

        best_match = get_best_fuzzy(value, self.channels)

        return super(MultiChannelProperty, self).__getitem__(self.channels.index(best_match))

    def __repr__(self):

        return '<MultiChannelProperty {0!r}, release={1!r}, channels={2!r}>'.format(
            self.name, self.parent.release if self.parent else None, self.channels)
