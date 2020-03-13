#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-10-25
# @Filename: base.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-07-25 18:32:35

from __future__ import absolute_import, division, print_function

import copy as copy_mod
import os

import astropy.table as table
from astropy import units as u

from marvin.core.exceptions import MarvinError
from marvin.utils.datamodel import DataModelList
from marvin.utils.general.structs import FuzzyList


class DRPCubeDataModel(object):
    """A class representing a DRP Cube datamodel.

    Parameters
    ----------
    release : str
        The DRP release this datamodel describes.
    datacubes : list
        A list of `.DataCube` instances that describe the datacubes in this
        datamodel.
    spectra : list
        A list of `.Spectrum` instances that describe the datacubes in this
        datamodel.
    aliases : list
        A list of aliases for this datamodel.
    bitmask : dict
        A dictionary of `~marvin.utils.general.maskbit.Maskbit` objects.
    qual_flag : str
        The name of the quality bitmask flag. Must not include the ``MANGA_``
        prefix.

    """

    def __init__(self, release, datacubes=[], spectra=[], aliases=[], bitmasks=None,
                 qual_flag='DRP3QUAL'):

        self.release = release
        self.aliases = aliases

        self.datacubes = DataCubeList(datacubes, parent=self)
        self.spectra = SpectrumList(spectra, parent=self)

        self.bitmasks = bitmasks if bitmasks is not None else {}
        self.qual_flag = qual_flag

    def __repr__(self):
        return ('<DRPCubeDataModel release={0!r}, n_datacubes={1}, n_spectra={2}>'
                .format(self.release, len(self.datacubes), len(self.spectra)))

    def copy(self):
        """Returns a copy of the datamodel."""

        return copy_mod.deepcopy(self)

    def __eq__(self, value):
        """Uses fuzzywuzzy to return the closest property match."""

        datacube_names = [datacube.name for datacube in self.datacubes]
        spectrum_names = [spectrum.name for spectrum in self.spectra]

        if value in datacube_names:
            return self.datacubes[datacube_names.index(value)]
        elif value in spectrum_names:
            return self.spectra[spectrum_names.index(value)]

        try:
            datacube_best_match = self.datacubes[value]
        except ValueError:
            datacube_best_match = None

        try:
            spectrum_best_match = self.spectra[value]
        except ValueError:
            spectrum_best_match = None

        if ((datacube_best_match is None and spectrum_best_match is None) or
                (datacube_best_match is not None and spectrum_best_match is not None)):
            raise ValueError('too ambiguous input {!r}'.format(value))
        elif datacube_best_match is not None:
            return datacube_best_match
        elif spectrum_best_match is not None:
            return spectrum_best_match

    def __contains__(self, value):

        try:
            match = self.__eq__(value)
            if match is None:
                return False
            else:
                return True
        except ValueError:
            return False

    def __getitem__(self, value):
        return self == value

    def to_rss(self):
        """Returns a copy with `.RSSDatamodel` objects instead of datacubes."""

        if isinstance(self, DRPRSSDataModel):
            raise ValueError('this is already a DRPRSSDataModel')

        copy_of_self = self.copy()
        delattr(copy_of_self, 'datacubes')

        # Chances the type and converts datacubes to RSSDatamodel
        copy_of_self.__class__ = DRPRSSDataModel
        copy_of_self.rss = self.datacubes.to_rss(copy_of_self)

        # Resets the parent of the copied spectra
        for spectrum in copy_of_self.spectra:
            spectrum.parent = copy_of_self

        return copy_of_self


class DRPRSSDataModel(DRPCubeDataModel):
    """A class representing a DRP RSS listdatamodel."""

    def __init__(self, release, rss=[], spectra=[], aliases=[], bitmasks=None):

        self.release = release
        self.aliases = aliases

        self.rss = RSSList(rss, parent=self)
        self.spectra = SpectrumList(spectra, parent=self)

        self.bitmasks = bitmasks if bitmasks is not None else {}

    def __repr__(self):
        return ('<DRPRSSDataModel release={0!r}, n_rss={1}, n_spectra={2}>'
                .format(self.release, len(self.rss), len(self.spectra)))


class DRPCubeDataModelList(DataModelList):
    """A dictionary of DRP Cube datamodels."""

    base = {'DRPCubeDataModel': DRPCubeDataModel}

    def copy(self):

        copy_of_self = super(DRPCubeDataModelList, self).copy()
        copy_of_self.__class__ = DRPRSSDataModelList

        return copy_of_self


class DRPRSSDataModelList(DataModelList):
    """A dictionary of DRP RSS datamodels."""

    base = {'DRPRSSDataModel': DRPRSSDataModel}


class DataCubeList(FuzzyList):
    """Creates a list containing models and their representation."""

    def __init__(self, the_list, parent=None):

        self.parent = parent

        super(DataCubeList, self).__init__([])

        for item in the_list:
            self.append(item, copy=True)

    def copy(self):
        """Returns a copy of the datamodel."""

        return copy_mod.deepcopy(self)

    def mapper(self, value):
        """Helper method for the fuzzy list to match on the datacube name."""

        return value.name

    def append(self, value, copy=True):
        """Appends with copy."""

        append_obj = value if copy is False else copy_mod.deepcopy(value)
        append_obj.parent = self.parent

        if isinstance(append_obj, DataCube):
            super(DataCubeList, self).append(append_obj)
        else:
            raise ValueError('invalid datacube of type {!r}'.format(type(append_obj)))

    def to_rss(self, new_parent):
        """Returns a copy of this list as an `.RSSList` object."""

        if isinstance(self, RSSList):
            raise ValueError('this is already an RSSList')

        # Copies selef and resets the type to RSSList
        copy_of_self = self.copy()
        copy_of_self.__class__ = RSSList
        copy_of_self.parent = new_parent

        # Replaces each datacube in itself with a RSSDatamodel
        for label in self.list_names():
            copy_of_self.remove(copy_of_self[label])
            copy_of_self.append(self[label].to_rss(new_parent))

        return copy_of_self

    def list_names(self):
        """Returns a list with the names of the datacubes in this list."""

        return [item.name for item in self]

    def to_table(self, pprint=False, description=False, max_width=1000):
        """Returns an astropy table with all the datacubes in this datamodel.

        Parameters:
            pprint (bool):
                Whether the table should be printed to screen using astropy's
                table pretty print.
            description (bool):
                If ``True``, an extra column with the description of the
                datacube will be added.
            max_width (int or None):
                A keyword to pass to ``astropy.table.Table.pprint()`` with the
                maximum width of the table, in characters.

        Returns:
            result (``astropy.table.Table``):
                If ``pprint=False``, returns an astropy table containing
                the name of the datacube, whether it has ``ivar`` or
                ``mask``, the units, and a description (if
                ``description=True``)..

        """

        datacube_table = table.Table(
            None, names=['name', 'ivar', 'mask', 'unit', 'description',
                         'db_table', 'db_column', 'fits_extension'],
            dtype=['S20', bool, bool, 'S20', 'S500', 'S20', 'S20', 'S20'])

        if self.parent:
            datacube_table.meta['release'] = self.parent.release

        for datacube in self:
            unit = datacube.unit.to_string()

            datacube_table.add_row((datacube.name,
                                    datacube.has_ivar(),
                                    datacube.has_mask(),
                                    unit,
                                    datacube.description,
                                    datacube.db_table,
                                    datacube.db_column(),
                                    datacube.fits_extension()))

        if not description:
            datacube_table.remove_column('description')

        if pprint:
            datacube_table.pprint(max_width=max_width, max_lines=1e6)
            return

        return datacube_table

    def write_csv(self, filename=None, path=None, overwrite=None, **kwargs):
        ''' Write the datamodel to a CSV '''

        release = self.parent.release.lower().replace('-', '')

        if not filename:
            filename = 'drpcubes_dm_{0}.csv'.format(release)

        if not path:
            path = os.path.join(os.getenv("MARVIN_DIR"), 'docs', 'sphinx', '_static')

        fullpath = os.path.join(path, filename)
        table = self.to_table(**kwargs)
        table.write(fullpath, format='csv', overwrite=overwrite)


class RSSList(DataCubeList):
    """Creates a list containing RSSDatamodel and their representation."""

    def append(self, value, copy=True):
        """Appends with copy."""

        append_obj = value if copy is False else copy_mod.deepcopy(value)
        append_obj.parent = self.parent

        if isinstance(append_obj, RSS):
            super(RSSList, self).append(append_obj)
        else:
            raise ValueError('invalid RSS of type {!r}'.format(type(append_obj)))

    def write_csv(self, filename=None, path=None, overwrite=None, **kwargs):
        """Write the datamodel to a CSV"""

        release = self.parent.release.lower().replace('-', '')

        if not filename:
            filename = 'drprss_dm_{0}.csv'.format(release)

        if not path:
            path = os.path.join(os.getenv('MARVIN_DIR'), 'docs', 'sphinx', '_static')

        fullpath = os.path.join(path, filename)
        table = self.to_table(**kwargs)
        table.write(fullpath, format='csv', overwrite=overwrite)


class DataCube(object):
    """Represents a extension in the DRP logcube file.

    Parameters:
        name (str):
            The datacube name. This is the internal name that Marvin will use
            for this datacube. It is different from the ``extension_name``
            parameter, which must be identical to the extension name of the
            datacube in the logcube file.
        extension_name (str):
            The FITS extension containing this datacube.
        extension_wave (str):
            The FITS extension containing the wavelength for this datacube.
        extension_ivar (str or None):
            The extension that contains the inverse variance associated with
            this datacube, if any.
        extension_mask (str or None):
            The extension that contains the mask associated with this
            datacube, if any.
        db_table (str):
            The DB table in which the datacube is stored. Defaults to
            ``spaxel``.
        unit (astropy unit or None):
            The unit for this datacube.
        scale (float):
            The scaling factor for the values of the datacube.
        formats (dict):
            A dictionary with formats that can be used to represent the
            datacube. Default ones are ``latex`` and ``string``.
        pixmask_flag : str
            The name of the pixmask flag. Should be the full name, including the
            ``MANGA_`` part.
        description (str):
            A description for the datacube.

    """

    def __init__(self, name, extension_name, extension_wave=None,
                 extension_ivar=None, extension_mask=None, db_table='spaxel',
                 unit=u.dimensionless_unscaled, scale=1, formats={},
                 pixmask_flag='MANGA_DRP3PIXMASK', description=''):

        self.name = name

        self._extension_name = extension_name
        self._extension_wave = extension_wave
        self._extension_ivar = extension_ivar
        self._extension_mask = extension_mask

        self.pixmask_flag = pixmask_flag

        self.db_table = db_table

        self._parent = None

        self.formats = formats

        self.description = description

        self.unit = u.CompositeUnit(scale, unit.bases, unit.powers)

    def copy(self):
        return copy_mod.deepcopy(self)

    def to_rss(self, new_parent):
        """Creates a copy of this datacube as a `.RSS` object."""

        if isinstance(self, RSS):
            raise ValueError('this is already a RSS datamodel object.')

        assert isinstance(new_parent, DRPRSSDataModel)
        copy_of_self = self.copy()
        copy_of_self.__class__ = RSS
        copy_of_self.parent = new_parent
        copy_of_self.db_table = 'rssfiber'

        return copy_of_self

    @property
    def parent(self):
        """Retrieves the parent."""

        return self._parent

    @parent.setter
    def parent(self, value):
        """Sets the parent."""

        assert isinstance(value, DRPCubeDataModel), 'parent must be a DRPCubeDataModel'

        self._parent = value

    def full(self):
        """Returns the name string."""

        return self._extension_name.lower()

    def has_ivar(self):
        """Returns True is the datacube has an ivar extension."""

        return self._extension_ivar is not None

    def has_mask(self):
        """Returns True is the datacube has an mask extension."""

        return self._extension_mask is not None

    def fits_extension(self, ext=None):
        """Returns the FITS extension name."""

        assert ext is None or ext in ['ivar', 'mask'], 'invalid extension'

        if ext is None:
            return self._extension_name.upper()

        elif ext == 'ivar':
            if not self.has_ivar():
                raise MarvinError('no ivar extension for datacube {0!r}'.format(self.full()))
            return self._extension_ivar.upper()

        elif ext == 'mask':
            if not self.has_mask():
                raise MarvinError('no mask extension for datacube {0!r}'.format(self.full()))
            return self._extension_mask

    def db_column(self, ext=None):
        """Returns the name of the DB column containing this datacube."""

        return self.fits_extension(ext=ext).lower()

    def __repr__(self):

        return '<DataCube {!r}, release={!r}, unit={!r}>'.format(
            self.name, self.parent.release if self.parent else None, self.unit.to_string())

    def __str__(self):

        return self.full()

    def to_string(self, mode='string'):
        """Return a string representation of the datacube."""

        if mode == 'latex':

            if mode in self.formats:
                latex = self.formats[mode]
            else:
                latex = self.to_string()

            return latex

        else:

            if mode in self.formats:
                string = self.formats[mode]
            else:
                string = self.name

            return string


class RSS(DataCube):

    def __repr__(self):

        return '<RSS {!r}, release={!r}, unit={!r}>'.format(
            self.name, self.parent.release if self.parent else None,
            self.unit.to_string())

    @property
    def parent(self):
        """Retrieves the parent."""

        return self._parent

    @parent.setter
    def parent(self, value):
        """Sets the parent."""

        assert isinstance(value, DRPRSSDataModel), 'parent must be a DRPRSSDataModel'
        self._parent = value


class SpectrumList(FuzzyList):
    """Creates a list containing spectra and their representation."""

    def __init__(self, the_list, parent=None):

        self.parent = parent

        super(SpectrumList, self).__init__([])

        for item in the_list:
            self.append(item, copy=True)

    def mapper(self, value):
        """Helper method for the fuzzy list to match on the spectrum name."""

        return value.name

    def append(self, value, copy=True):
        """Appends with copy."""

        append_obj = value if copy is False else copy_mod.deepcopy(value)
        append_obj.parent = self.parent

        if isinstance(append_obj, Spectrum):
            super(SpectrumList, self).append(append_obj)
        else:
            raise ValueError('invalid spectrum of type {!r}'.format(type(append_obj)))

    def list_names(self):
        """Returns a list with the names of the spectra in this list."""

        return [item.name for item in self]

    def to_table(self, pprint=False, description=False, max_width=1000):
        """Returns an astropy table with all the spectra in this datamodel.

        Parameters:
            pprint (bool):
                Whether the table should be printed to screen using astropy's
                table pretty print.
            description (bool):
                If ``True``, an extra column with the description of the
                spectrum will be added.
            max_width (int or None):
                A keyword to pass to ``astropy.table.Table.pprint()`` with the
                maximum width of the table, in characters.

        Returns:
            result (``astropy.table.Table``):
                If ``pprint=False``, returns an astropy table containing
                the name of the spectrum, whether it has ``ivar`` or
                ``mask``, the units, and a description (if
                ``description=True``)..

        """

        spectrum_table = table.Table(
            None, names=['name', 'std', 'unit', 'description',
                         'db_table', 'db_column', 'fits_extension'],
            dtype=['S20', bool, 'S20', 'S500', 'S20', 'S20', 'S20'])

        if self.parent:
            spectrum_table.meta['release'] = self.parent.release

        for spectrum in self:
            unit = spectrum.unit.to_string()

            spectrum_table.add_row((spectrum.name,
                                    spectrum.has_std(),
                                    unit,
                                    spectrum.description,
                                    spectrum.db_table,
                                    spectrum.db_column(),
                                    spectrum.fits_extension()))

        if not description:
            spectrum_table.remove_column('description')

        if pprint:
            spectrum_table.pprint(max_width=max_width, max_lines=1e6)
            return

        return spectrum_table

    def write_csv(self, filename=None, path=None, overwrite=None, **kwargs):
        ''' Write the datamodel to a CSV '''

        release = self.parent.release.lower().replace('-', '')

        if not filename:
            if isinstance(self.parent, DRPRSSDataModel):
                filename = 'drp_rss_spectra_dm_{0}.csv'.format(release)
            elif isinstance(self.parent, DRPCubeDataModel):
                filename = 'drp_cube_spectra_dm_{0}.csv'.format(release)
            else:
                raise ValueError('invalid parent of type {!r}'.format(type(self.parent)))

        if not path:
            path = os.path.join(os.getenv("MARVIN_DIR"), 'docs', 'sphinx', '_static')

        fullpath = os.path.join(path, filename)
        table = self.to_table(**kwargs)
        table.write(fullpath, format='csv', overwrite=overwrite)


class Spectrum(object):
    """Represents a extension in the DRP logcube file.

    Parameters:
        name (str):
            The spectrum name. This is the internal name that Marvin will use
            for this spectrum. It is different from the ``extension_name``
            parameter, which must be identical to the extension name of the
            spectrum in the logcube file.
        extension_name (str):
            The FITS extension containing this spectrum.
        extension_wave (str):
            The FITS extension containing the wavelength for this spectrum.
        extension_std (str):
            The FITS extension containing the standard deviation for this
            spectrum.
        extension_mask (str):
            The FITS extension containing the mask for this spectrum.
        db_table (str):
            The DB table in which the spectrum is stored. Defaults to
            ``cube``.
        unit (astropy unit or None):
            The unit for this spectrum.
        scale (float):
            The scaling factor for the values of the spectrum.
        formats (dict):
            A dictionary with formats that can be used to represent the
            spectrum. Default ones are ``latex`` and ``string``.
        pixmask_flag : str
            The name of the pixmask flag. Should be the full name, including the
            ``MANGA_`` part.
        description (str):
            A description for the spectrum.

    """

    def __init__(self, name, extension_name, extension_wave=None, extension_std=None,
                 extension_mask=None, db_table='cube', unit=u.dimensionless_unscaled,
                 scale=1, formats={}, pixmask_flag=None, description=''):

        self.name = name

        self._extension_name = extension_name
        self._extension_wave = extension_wave
        self._extension_std = extension_std
        self._extension_mask = extension_mask

        self.pixmask_flag = pixmask_flag

        self.db_table = db_table

        self.formats = formats

        self.description = description

        self._parent = None

        self.unit = u.CompositeUnit(scale, unit.bases, unit.powers)

    @property
    def parent(self):
        """Retrieves the parent."""

        return self._parent

    @parent.setter
    def parent(self, value):
        """Sets the parent."""

        assert isinstance(value, DRPCubeDataModel), 'parent must be a DRPCubeDataModel'

        self._parent = value

    def full(self):
        """Returns the name string."""

        return self._extension_name.lower()

    def has_std(self):
        """Returns True is the datacube has an std extension."""

        return self._extension_std is not None

    def has_mask(self):
        """Returns True is the datacube has an mask extension."""

        return self._extension_mask is not None

    def fits_extension(self, ext=None):
        """Returns the FITS extension name."""

        assert ext is None or ext in ['std', 'mask'], 'invalid extension'

        if ext is None:
            return self._extension_name.upper()

        elif ext == 'std':
            if not self.has_std():
                raise MarvinError('no std extension for spectrum {0!r}'.format(self.full()))
            return self._extension_std.upper()

        elif ext == 'mask':
            if not self.has_mask():
                raise MarvinError('no mask extension for spectrum {0!r}'.format(self.full()))
            return self._extension_mask

    def db_column(self, ext=None):
        """Returns the name of the DB column containing this datacube."""

        return self.fits_extension(ext=ext).lower()

    def __repr__(self):

        return '<Spectrum {!r}, release={!r}, unit={!r}>'.format(
            self.name, self.parent.release if self.parent else None, self.unit.to_string())

    def __str__(self):

        return self.full()

    def to_string(self, mode='string'):
        """Return a string representation of the spectrum."""

        if mode == 'latex':

            if mode in self.formats:
                latex = self.formats[mode]
            else:
                latex = self.to_string()

            return latex

        else:

            if mode in self.formats:
                string = self.formats[mode]
            else:
                string = self.name

            return string
