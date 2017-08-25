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
import re
import six

from collections import OrderedDict

import astropy.table as table
from astropy import units as u

from marvin.core.exceptions import MarvinError, MarvinNotImplemented
from marvin.utils.general.structs import get_best_fuzzy


__ALL__ = ('DAPDataModelList', 'DAPDataModel', 'Bintype', 'Template', 'Property',
           'MultiChannelProperty', 'spaxel', 'datamodel', 'Channel', 'Bit'
           'Maskbit')


spaxel = u.Unit('spaxel', represents=u.pixel, doc='A spectral pixel', parse_strict='silent')


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

    def __init__(self, release, bintypes=[], templates=[], properties=[], bitmasks={},
                 default_template=None, default_bintype=None, aliases=[]):

        self.release = release
        self.bintypes = bintypes
        self.templates = templates

        self.aliases = aliases

        self._properties = []
        self.add_properties(properties, copy=True)

        self.bitmasks = bitmasks

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
    def extensions(self):
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

        prop_names = [prop.name for prop in self.extensions]

        # If the values matches exactly one of the property names, we return the property object.
        if value in prop_names:
            return self.extensions[prop_names.index(value)]

        # Finds the best property+channel
        propety_channel_names = self.list_property_names()

        try:
            best = get_best_fuzzy(value, propety_channel_names)
        except ValueError:
            # Second pass, removes _
            best = get_best_fuzzy(value, [pcn.replace('_', ' ') for pcn in propety_channel_names])
            best = best.replace(' ', '_')

        return self.properties[propety_channel_names.index(best)]

    @property
    def default_bintype(self):
        """Returns the default bintype."""

        return self._default_bintype

    @default_bintype.setter
    def default_bintype(self, value):
        """Sets the default bintype."""

        for bintype in self.bintypes:
            if str(value) == bintype.name:
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

        for template in self.templates:
            if str(value) == template.name:
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

        for bintype in self.bintypes:
            if bintype.name.upper() == str(value).upper():
                return bintype

        raise MarvinError('invalid bintype {0!r}'.format(value))

    def get_template(self, value=None):
        """Returns a template whose name matches ``value``.

        Returns the default one if ``value=None``.

        """

        if value is None:
            return self.default_template

        for template in self.templates:
            if template.name.upper() == str(value).upper():
                return template

        raise MarvinError('invalid template {0!r}'.format(value))

    def get_bintemps(self, default=False):
        """Returns a list of all combinations of bintype and template."""

        if default:
            return '{0}-{1}'.format(self.default_bintype.name, self.default_template.name)

        bins = [bintype.name for bintype in self.bintypes]
        temps = [template.name for template in self.templates]

        return ['-'.join(item) for item in list(itertools.product(bins, temps))]

    def to_table(self, compact=True, pprint=False, description=False, max_width=1000):
        """Returns an astropy table with all the properties in this model.

        Parameters:
            compact (bool):
                If ``True``, groups extensions (multichannel properties) in one
                line. Otherwise, shows a row for each extension and channel.
            pprint (bool):
                Whether the table should be printed to screen using astropy's
                table pretty print.
            description (bool):
                If ``True``, an extra column with the description of the
                property will be added.
            max_width (int or None):
                A keyword to pass to ``astropy.table.Table.pprint()`` with the
                maximum width of the table, in characters.

        Returns:
            result (``astropy.table.Table``):
                If ``pprint=False``, returns an astropy table containing
                the name of the property, the channel (or channels, if
                ``compact=True``), whether the property has ``ivar`` or
                ``mask``, the units, and a description (if
                ``description=True``). Additonal information such as the
                bintypes, templates, release, etc. is included in
                the metadata of the table (use ``.meta`` to access them).

        """

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
            iterable = self.extensions
        else:
            iterable = self.properties

        for prop in iterable:
            if isinstance(prop, MultiChannelProperty):
                channel = ', '.join([ch.name for ch in prop.channels])
                units = [pp.unit.to_string() for pp in prop]
                unit = units[0] if len(set(units)) == 1 else 'multiple'
            else:
                channel = '' if not prop.channel else prop.channel
                unit = prop.unit.to_string()

            model_table.add_row((prop.name, channel, prop.ivar, prop.mask, unit, prop.description))

        if not description:
            model_table.remove_column('description')

        if pprint:
            model_table.pprint(max_width=max_width, max_lines=1e6)
            return

        return model_table


class Bintype(object):
    """A class representing a bintype."""

    def __init__(self, bintype, binned=True, n=None, description=''):

        self.name = bintype
        self.binned = binned
        self.n = n
        self.description = description

    def __repr__(self):

        return '<Bintype {0!r}, binned={1!r}>'.format(self.name, self.binned)

    def __str__(self):

        return self.name


class Template(object):
    """A class representing a template."""

    def __init__(self, template, n=None, description=''):

        self.name = template
        self.n = n
        self.description = description

    def __repr__(self):

        return '<Template {0!r}>'.format(self.name)

    def __str__(self):

        return self.name


class Property(object):
    """A class representing a DAP property.

    Parameters:
        name (str):
            The name of the property.
        channel (:class:`Channel` object or None):
            The channel associated to the property, if any.
        ivar (bool):
            Whether the property has an inverse variance measurement.
        mask (bool):
            Whether the property has an associated mask.
        unit (astropy unit or None):
            The unit for this channel. If not defined, the unit from the
            ``channel`` will be used.
        scale (float or None):
            The scaling factor for the property. If not defined, the scaling
            factor from the ``channel`` will be used.
        formats (dict):
            A dictionary with formats that can be used to represent the
            property. Default ones are ``latex`` and ``string``.
        parent (:class:`DAPDataModel` object or None):
            The associated :class:`DAPDataModel` object. Usually it is set to
            ``None`` and populated when the property is added to the
            ``DAPDataModel`` object.
        binid (:class:`Property` object or None):
            The ``binid`` :class:`Property` object associated with this
            propety. Useful in case of hybrid binning.
        description (str):
            A description of the property.

    """

    def __init__(self, name, channel=None, ivar=False, mask=False, unit=u.dimensionless_unscaled,
                 scale=1, formats={}, parent=None, binid=None, description=''):

        self.name = name
        self.channel = channel

        self.ivar = ivar
        self.mask = mask
        self.binid = binid

        self.formats = formats

        if self.channel is None:
            self.scale = scale
            self.unit = unit
        else:
            self.scale = scale if scale is not None else self.channel.scale
            self.unit = unit if unit is not None else self.channel.unit

        # Makes sure the channel shares the units and scale
        if self.channel:
            self.channel.scale = self.scale
            self.channel.unit = self.unit

        self.description = description

        self.parent = parent

    def set_parent(self, parent):
        """Sets parent."""

        assert isinstance(parent, DAPDataModel), 'parent must be a DAPDataModel'

        self.parent = parent

    def full(self):
        """Returns the name + channel string."""

        if self.channel:
            return self.name + '_' + self.channel.name

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
            return self.name + '_ivar' + ('_{0}'.format(self.channel.name) if self.channel else '')

        if ext == 'mask':
            assert self.mask is True, 'no mask for property {0!r}'.format(self.full())
            return self.name + '_mask' + ('_{0}'.format(self.channel.name) if self.channel else '')

    def __repr__(self):

        return '<Property {0!r}, release={1!r}, channel={2!r}, unit={3!r}>'.format(
            self.name, self.parent.release if self.parent else None,
            self.channel.name if self.channel else 'None', self.unit.to_string())

    def __str__(self):

        return self.full()

    def to_string(self, mode='string', include_channel=True):
        """Return a string representation of the channel."""

        if mode == 'latex':

            if mode in self.formats:
                latex = self.formats[mode]
            else:
                latex = self.to_string(include_channel=False).replace(' ', '\\ ')

            if self.channel and include_channel:
                latex = latex + ':\\ ' + self.channel.to_string('latex')

            return latex

        else:

            if mode in self.formats:
                string = self.formats[mode]
            else:
                string = self.name

            if self.channel is None or include_channel is False:
                return string
            else:
                return string + ': ' + self.channel.to_string(mode=mode)


class MultiChannelProperty(list):
    """A class representing a list of channels for the same property.

    Parameters:
        name (str):
            The name of the property.
        channels (list of :class:`Channel` objects):
            The channels associated to the property.
        ivar (bool):
            Whether the properties have an inverse variance measurement.
        mask (bool):
            Whether the properties have an associated mask.
        unit (astropy unit or None):
            The unit for these channels. If set, it will override any unit
            defined in the individual channels.
        scale (float):
            The scaling factor for these channels. If set, it will override
            any unit defined in the individual channels.
        formats (dict):
            A dictionary with formats that can be used to represent the
            property. Default ones are ``latex`` and ``string``.
        parent (:class:`DAPDataModel` object or None):
            The associated :class:`DAPDataModel` object. Usually it is set to
            ``None`` and populated when the property is added to the
            ``DAPDataModel`` object.
        description (str):
            A description of the property.
        kwargs (dict):
            Arguments to be passed to each ``Property`` on initialisation.

    """

    def __init__(self, name, channels=[], unit=None, scale=None, **kwargs):

        self.name = name
        self.channels = channels

        self.ivar = kwargs.get('ivar', False)
        self.mask = kwargs.get('mask', False)
        self.description = kwargs.get('description', '')

        self.parent = kwargs.get('parent', None)

        self_list = []
        for ii, channel in enumerate(channels):
            this_unit = unit if not isinstance(unit, (list, tuple)) else unit[ii]
            self_list.append(Property(self.name, channel=channel,
                                      unit=this_unit, scale=scale, **kwargs))

        list.__init__(self, self_list)

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
            self.name, self.parent.release if self.parent else None,
            [channel.name for channel in self.channels])


class Channel(object):
    """A class representing a channel in a property.

    Parameters:
        name (str):
            The channel name.
        unit (astropy unit or None):
            The unit for this channel.
        scale (float):
            The scaling factor for the channel.
        formats (dict):
            A dictionary with formats that can be used to represent the
            channel. Default ones are ``latex`` and ``string``.
        idx (int):
            The index of the channel in the MAPS file extension.
        description (str):
            A description for the channel.

    """

    def __init__(self, name, unit=u.dimensionless_unscaled, scale=1, formats={}, idx=None,
                 description=''):

        self.name = name
        self.unit = unit
        self.scale = scale
        self.formats = formats
        self.idx = idx
        self.description = description

    def to_string(self, mode='string'):
        """Return a string representation of the channel."""

        if mode == 'latex':
            if 'latex' in self.formats:
                latex = self.formats['latex']
                latex = re.sub(r'forb{(.+)}', r'lbrack\\textrm{\1}\\rbrack', latex)
            else:
                latex = self.to_string().replace(' ', '\\ ')
            return latex
        elif mode is not None and mode in self.formats:
            return self.formats[mode]
        else:
            return self.name

    def __repr__(self):

        return '<Channel {0!r} unit={1!r}>'.format(self.name, self.unit.to_string())

    def __str__(self):

        return self.name

class Bit(object):
    """A class representing a single bit of a maskbit.

    Parameters:
        value (int):
            Value of .
        name (str):
            Name of bit.
        description (str):
            Description of bit.
    """

    def __init__(self, value, name, description):

        self.value = value
        self.name = name
        self.description = description

    def __repr__(self):

        return '<Bit {0:>2} name={1!r}>'.format(self.value, self.name)


class Maskbit(object):
    """A class representing a maskbit.
    
    Parameters:
        bits (OrderedDict):
            Collection of ``marvin.utils.dap.datamodel.base.Bit`` objects.
        name (str):
            Name of maskbit.
        description (str):
            Description of maskbit.
    """

    def __init__(self, bits, name, description):

        self.bits = bits
        self.name = name
        self.description = description

    def __repr__(self):

        return '<Maskbit name={0!r}>'.format(self.name)

    def to_table(self, pprint=False, description=True, max_width=1000):
        """Returns an astropy table with all of the bits.

        Parameters:
            pprint (bool):
                Whether the table should be printed to screen using
                astropy's table pretty print.
            description (bool):
                If ``True``, an extra column with the description of
                the property will be added. Default is ``True``.
            max_width (int or None):
                A keyword to pass to ``astropy.table.Table.pprint()``
                with the maximum width of the table, in characters.

        Returns:
            result (``astropy.table.Table``):
                If ``pprint=False``, returns an astropy table
                containing the value and names of the bits.
        """

        model_table = table.Table(None, names=['bit', 'name', 'description'],
                                  dtype=['i2', 'S30', 'S500'])

        for bit in self.bits.values():
            model_table.add_row((bit.value, bit.name, bit.description))

        if not description:
            model_table.remove_column('description')

        if pprint:
            model_table.pprint(max_width=max_width, max_lines=1e6)
            return

        return model_table
