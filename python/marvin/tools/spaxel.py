#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Nov 3, 2017
# @Filename: spaxel.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import abc
import itertools
import six
import warnings

import numpy as np

import marvin
import marvin.core.core
import marvin.core.exceptions
import marvin.core.marvin_pickle
import marvin.utils.general.general

import marvin.tools.cube
import marvin.tools.maps
import marvin.tools.modelcube

from marvin.utils.general.structs import FuzzyDict

from marvin.core.exceptions import MarvinError, MarvinUserWarning, MarvinBreadCrumb

from marvin.utils.datamodel.dap import datamodel as dap_datamodel
from marvin.utils.datamodel.drp import datamodel as drp_datamodel


breadcrumb = MarvinBreadCrumb()


def spaxel_factory(cls, *args, **kwargs):
    """A factory that returns the right type of spaxel depending on input.

    Based on the input values, determines if the resulting spaxels should be
    binned or unbinned, returning a `.Spaxel` or a `.Bin` respectively.
    This function is intended for overrding the ``__call__`` method in the
    `abc.ABCMeta` metacalass. The reason is that we want `.SpaxelBase` to have
    `abstract methods <abc.abstractmethod>` while also being a factory.
    See `this stack overflow <https://stackoverflow.com/a/5961102>`_ for
    details in the implementation of the ``__call__`` factory pattern.

    It can be used as::

        SpaxelABC = abc.ABCMeta
        SpaxelABC.__call__ = region_factory


        class SpaxelBase(object, metaclass=RegionABC):
            ...

    Note that this will override ``__call__`` everywhere else where
    `abc.ABCMeta` is used, but since it only overrides the default behaviour
    when the input class is `.SpaxelBase`, that should not be a problem.

    """

    if cls is not SpaxelBase:

        return type.__call__(cls, *args, **kwargs)

    bintype = kwargs.get('bintype', None)

    release = kwargs.get('release', marvin.config.release)
    mode = kwargs.get('mode', marvin.config.mode)
    download = kwargs.get('download', marvin.config.download)

    maps_filename = kwargs.get('maps_filename', None)
    modelcube_filename = kwargs.get('modelcube_filename', None)

    maps = kwargs.get('maps', None)
    modelcube = kwargs.get('modelcube', None)

    # If we are not going to load maps or modelcube information, returns a Spaxel.
    if (maps_filename is None and modelcube_filename is None and
            (maps is False or maps is None) and (modelcube is False or modelcube is None)):
        return Spaxel(*args, **kwargs)

    # If one of maps or modelcube is an object, uses it to determine the bin type.
    # If there is any inconsistency with the filenames or between each other, it will
    # cause an error when initialising the Spaxel/Bin.
    if isinstance(maps, marvin.tools.maps.Maps):
        if maps.is_binned() is False:
            return Spaxel(*args, **kwargs)
        else:
            return Bin(*args, **kwargs)
    elif isinstance(modelcube, marvin.tools.modelcube.ModelCube):
        if modelcube.is_binned() is False:
            return Spaxel(*args, **kwargs)
        else:
            return Bin(*args, **kwargs)

    # First we check the case in which filename are not set. That means that
    # we will be using the bintype keyword. We use the datamodel to determine
    # whether it is binned or not.
    if maps_filename is None and modelcube_filename is None:
        datamodel = dap_datamodel[release]
        bintype_dm = datamodel.get_bintype(bintype)
        if bintype_dm.binned is False:
            return Spaxel(*args, **kwargs)
        else:
            return Bin(*args, **kwargs)

    # Last chance is that one of the filename are not null. We instantiate the
    # file to determine the bintype
    if maps_filename is not None:
        maps = marvin.tools.maps.Maps(filename=maps_filename,
                                      release=release, mode=mode, download=download)
        kwargs.update(maps=maps, maps_filename=None)
        if maps.is_binned() is False:
            return Spaxel(*args, **kwargs)
        else:
            return Bin(*args, **kwargs)

    if modelcube_filename is not None:
        modelcube = marvin.tools.modelcube.Maps(filename=modelcube_filename,
                                                release=release, mode=mode, download=download)
        kwargs.update(modelcube=modelcube, modelcube_filename=None)
        if modelcube.is_binned() is False:
            return Spaxel(*args, **kwargs)
        else:
            return Bin(*args, **kwargs)

    raise MarvinError('you have reached the end of the SpaxelBase logic. '
                      'This should never happen!')


# Overrides the __call__ method in abc.ABC.
SpaxelABC = abc.ABCMeta
SpaxelABC.__call__ = spaxel_factory


class DataModel(object):
    """A single ibject that holds the DRP and DAP datamodel."""

    def __init__(self, release):

        self.drp = drp_datamodel[release]
        self.dap = dap_datamodel[release]


class SpaxelBase(six.with_metaclass(SpaxelABC, object)):
    """A base class that contains information about a spaxel.

    This class represents an spaxel with information from the reduced DRP
    spectrum, the DAP maps properties, and the model spectrum from the DAP
    logcube. A `.SpaxelBase` can be initialised with all or only part of that
    information, and either from a file, a database, or remotely via the
    Marvin API.

    The `~marvin.tools.cube.Cube`, `~marvin.tools.maps.Maps` , and
    `~marvin.tools.modelcube.ModelCube` quantities for the spaxel are available
    in ``cube_quantities``, ``maps_quantities``, and ``modelcube_quantities``,
    respectively. For convenience, the quantities can also be accessed directly
    from the `.SpaxelBase` itself (e.g., ``spaxel.emline_gflux_ha_6465``).

    Parameters:
        x,y (int):
            The `x` and `y` coordinates of the spaxel in the cube (0-indexed).
        cube_filename (str):
            The path of the data cube file containing the spaxel to load.
        maps_filename (str):
            The path of the DAP MAPS file containing the spaxel to load.
        modelcube_filename (str):
            The path of the DAP model cube file containing the spaxel to load.
        mangaid (str):
            The mangaid of the cube/maps of the spaxel to load.
        plateifu (str):
            The plate-ifu of the cube/maps of the spaxel to load (either
            ``mangaid`` or ``plateifu`` can be used, but not both).
        cube (`~marvin.tools.cube.Cube` object or bool):
            If ``cube`` is a `~marvin.tools.cube.Cube` object, that
            cube will be used for the `.SpaxelBase` instantiation. This mode
            is mostly intended for `~marvin.utils.general.general.getSpaxel`
            as it significantly improves loading time. Otherwise, ``cube`` can
            be ``True`` (default), in which case a cube will be instantiated
            using the input ``filename``, ``mangaid``, or ``plateifu``. If
            ``cube=False``, no cube will be used and the cube associated
            quantities will not be available..
        maps (`~marvin.tools.maps.Maps` object or bool)
            As ``cube`` but for the DAP measurements corresponding to the
            spaxel in the `.Maps`.
        modelcube (`marvin.tools.modelcube.ModelCube` object or bool)
            As ``maps`` but for the DAP measurements corresponding to the
            spaxel in the `.ModelCube`.
        bintype (str or None):
            The binning type. For MPL-4, one of the following: ``'NONE',
            'RADIAL', 'STON'`` (if ``None`` defaults to ``'NONE'``).
            For MPL-5, one of, ``'ALL', 'NRE', 'SPX', 'VOR10'``
            (defaults to ``'SPX'``). MPL-6 also accepts the ``'HYB10'`` binning
            schema.
        template (str or None):
            The template use for kinematics. For MPL-4, one of
            ``'M11-STELIB-ZSOL', 'MILES-THIN', 'MIUSCAT-THIN'`` (if ``None``,
            defaults to ``'MIUSCAT-THIN'``). For MPL-5 and successive, the only
            option in ``'GAU-MILESHC'`` (``None`` defaults to it).
        release (str):
            The MPL/DR version of the data to use.
        lazy (bool):
            If ``False``, the spaxel data is loaded on instantiation.
            Otherwise, only the metadata is created. The associated quantities
            can be then loaded by calling `.SpaxelBase.load()`.
        kwargs (dict):
            Arguments to be passed to `.Cube`, `.Maps`, and `.ModelCube`
            when (and if) they are initialised.

    Attributes:
        cube_quantities (`~marvin.utils.general.structs.FuzzyDict`):
            A querable dictionary with the `.Spectrum` quantities
            derived from `.Cube` and matching ``x, y``.
        datamodel (object):
            An object contianing the DRP and DAP datamodels.
        maps_quantities (`~marvin.utils.general.structs.FuzzyDict`):
            A querable dictionary with the `.AnalysisProperty` quantities
            derived from `.Maps` and matching ``x, y``.
        model_quantities (`~marvin.utils.general.structs.FuzzyDict`):
            A querable dictionary with the `.Spectrum` quantities
            derived from `.ModelCube` and matching ``x, y``.
        ra,dec (float):
            Right ascension and declination of the spaxel. Not available until
            the spaxel has been `loaded <.SpaxelBase.load>`.

    """

    def __init__(self, x, y, cube_filename=None, maps_filename=None,
                 modelcube_filename=None, mangaid=None, plateifu=None,
                 cube=True, maps=True, modelcube=True, bintype=None,
                 template=None, template_kin=None, release=None, lazy=False,
                 **kwargs):

        if template_kin is not None:
            warnings.warn('template_kin has been deprecated and will be removed '
                          'in a future version. Use template.',
                          marvin.core.exceptions.MarvinDeprecationWarning)
            template = template_kin

        self.cube = cube
        self.maps = maps
        self.modelcube = modelcube

        if not self.cube and not self.maps and not self.modelcube:
            raise MarvinError('either cube, maps, or modelcube must be True or '
                              'a Marvin Cube, Maps, or ModelCube object must be specified.')

        self.plateifu = self._check_versions('plateifu', plateifu)
        self.mangaid = self._check_versions('mangaid', mangaid)

        self._parent_shape = None

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinSpaxel {0}'.format(self.__class__),
                        category=self.__class__)

        # Checks versions
        input_release = release if release is not None else marvin.config.release
        self.release = self._check_versions('release', input_release, check_input=False)
        assert self.release in marvin.config._mpldict, 'invalid release version.'

        self.x = int(x)
        self.y = int(y)
        assert self.x is not None and self.y is not None, 'Spaxel requires x and y to initialise.'

        self.loaded = False
        self.datamodel = DataModel(self.release)

        self.bintype = self.datamodel.dap.get_bintype(self._check_versions('bintype', bintype))
        self.template = self.datamodel.dap.get_template(self._check_versions('template', template))

        self.cube_quantities = FuzzyDict({})
        self.maps_quantities = FuzzyDict({})
        self.modelcube_quantities = FuzzyDict({})

        # Stores the remaining input values to be used with load()
        self.__input_params = dict(cube_filename=cube_filename,
                                   maps_filename=maps_filename,
                                   modelcube_filename=modelcube_filename,
                                   kwargs=kwargs)

        if lazy is False:
            self.load()

    def __dir__(self):

        items = self.cube_quantities.__dir__()
        items += self.maps_quantities.__dir__()
        items += self.modelcube_quantities.__dir__()
        items += super(SpaxelBase, self).__dir__()

        return sorted(items)

    def __getattr__(self, value):

        if value in self.cube_quantities:
            return self.cube_quantities[value]
        if value in self.maps_quantities:
            return self.maps_quantities[value]
        if value in self.modelcube_quantities:
            return self.modelcube_quantities[value]

        return super(SpaxelBase, self).__getattribute__(value)

    @abc.abstractmethod
    def __repr__(self):

        return '<SpaxelBase>'

    def _check_versions(self, attr, input_value, check_input=True):
        """Checks that all input object have the same versions.

        Runs sanity checks to make sure that ``attr`` has the same value
        in the input `.Cube`, `.Maps`, and `.ModelCube`. If
        ``check_input=True``, also checks that the ``input_value`` for the
        attribute matches the ones in the Marvin objects.

        Returns the value for the attribute based on the input value and the
        Marvin objects, or raises an error if there is an inconsistency.

        """

        inputs = []
        for obj in [self.cube, self.maps, self.modelcube]:
            if obj is not None and not isinstance(obj, bool):
                inputs.append(obj)

        if len(inputs) == 1:
            if not hasattr(inputs[0], attr):
                return input_value
            if input_value is not None:
                if input_value is not None and check_input:
                    assert input_value == getattr(inputs[0], attr), \
                        'input {!r} does not match {!r}'.format(attr, inputs[0])
            return getattr(inputs[0], attr)

        output_value = input_value

        for obj_a, obj_b in itertools.combinations(inputs, 2):
            if hasattr(obj_a, attr) and hasattr(obj_b, attr):
                assert getattr(obj_a, attr) == getattr(obj_b, attr)
                if input_value is not None and check_input:
                    assert input_value == getattr(obj_a, attr), \
                        'input {!r} does not match {!r}'.format(attr, obj_a)
                output_value = getattr(obj_a, attr)

        return output_value

    def _set_radec(self):
        """Calculates ra and dec for this spaxel."""

        self.ra = None
        self.dec = None

        for obj in [self.cube, self.maps, self.modelcube]:
            if hasattr(obj, 'wcs'):
                if obj.wcs.naxis == 2:
                    self.ra, self.dec = obj.wcs.wcs_pix2world([[self.x, self.y]], 0)[0]
                elif obj.wcs.naxis == 3:
                    self.ra, self.dec, __ = obj.wcs.wcs_pix2world([[self.x, self.y, 0]], 0)[0]

    def load(self):
        """Loads the spaxel data."""

        if self.loaded:
            warnings.warn('already loaded', MarvinUserWarning)
            return

        self._load_cube()
        self._load_maps()
        self._load_modelcube()

        self._set_radec()
        self.loaded = True

    def save(self, path, overwrite=False):
        """Pickles the spaxel to a file.

        Parameters:
            path (str):
                The path of the file to which the `.Spaxel` will be saved.
                Unlike for other Marvin Tools that derive from
                `~marvin.core.core.MarvinToolsClass`, ``path`` is
                mandatory for `.Spaxel.save` as there is no default path for a
                given spaxel.
            overwrite (bool):
                If True, and the ``path`` already exists, overwrites it.
                Otherwise it will fail.

        Returns:
            path (str):
                The realpath to which the file has been saved.

        """

        return marvin.core.marvin_pickle.save(self, path=path, overwrite=overwrite)

    @classmethod
    def restore(cls, path, delete=False):
        """Restores a Spaxel object from a pickled file.

        If ``delete=True``, the pickled file will be removed after it has been
        unplickled. Note that, for objects with ``data_origin='file'``, the
        original file must exists and be in the same path as when the object
        was first created.

        """

        return marvin.core.marvin_pickle.restore(path, delete=delete)

    def _load_cube(self):
        """Loads the cube and the associated quantities."""

        # Checks that the cube is correct or load ones if cube == True.
        if not isinstance(self.cube, bool):

            assert isinstance(self.cube, marvin.tools.cube.Cube), \
                'cube is not an instance of marvin.tools.cube.Cube or a boolean.'

        elif self.cube is True:

            self.cube = marvin.tools.cube.Cube(filename=self.__input_params['cube_filename'],
                                               plateifu=self.plateifu,
                                               mangaid=self.mangaid,
                                               release=self.release,
                                               **self.__input_params['kwargs'])

        else:

            self.cube = None
            return

        self._parent_shape = self.cube._shape

        self.cube_quantities = self.cube._get_spaxel_quantities(self.x, self.y)

    def _load_maps(self):
        """Loads the cube and the properties."""

        if not isinstance(self.maps, bool):

            assert isinstance(self.maps, marvin.tools.maps.Maps), \
                'maps is not an instance of marvin.tools.maps.Maps or a boolean.'

        elif self.maps is True:

            self.maps = marvin.tools.maps.Maps(filename=self.__input_params['maps_filename'],
                                               mangaid=self.mangaid,
                                               plateifu=self.plateifu,
                                               bintype=self.bintype,
                                               template=self.template,
                                               release=self.release,
                                               **self.__input_params['kwargs'])

        else:

            self.maps = None
            return

        self._parent_shape = self.maps._shape

        self.maps_quantities = self.maps._get_spaxel_quantities(self.x, self.y)

    def _load_modelcube(self):
        """Loads the modelcube and associated arrays."""

        if not isinstance(self.modelcube, bool):

            assert isinstance(self.modelcube, marvin.tools.modelcube.ModelCube), \
                'modelcube is not an instance of marvin.tools.modelcube.ModelCube or a boolean.'

        elif self.modelcube is True:

            if self.release == 'MPL-4':
                warnings.warn('ModelCube cannot be instantiated for MPL-4.', MarvinUserWarning)
                self.modelcube = None
                return

            self.modelcube = marvin.tools.modelcube.ModelCube(
                filename=self.__input_params['modelcube_filename'],
                mangaid=self.mangaid,
                plateifu=self.plateifu,
                bintype=self.bintype,
                template=self.template,
                release=self.release,
                **self.__input_params['kwargs'])

        else:

            self.modelcube = None
            return

        self._parent_shape = self.modelcube._shape

        self.modelcube_quantities = self.modelcube._get_spaxel_quantities(self.x, self.y)


class Spaxel(SpaxelBase):
    """A class representing an unbinned spaxel.

    This subclass of `.SpaxelBase` represents an spaxel belonging to an
    unbinned `~marvin.tools.maps.Maps` and `~marvin.tools.modelcube.ModelCube`.
    If initialised directly, a `.Spaxel` will fail if the input data correspond
    to a binned maps or modelcube object.

    Refer to the documentation of `.SpaxelBase` for information about the valid
    parameters and methods.

    """

    def __init__(self, *args, **kwargs):

        super(Spaxel, self).__init__(*args, **kwargs)

        if isinstance(self.maps, marvin.tools.maps.Maps):
            assert self.maps.is_binned() is False, 'a Spaxel cannot be binned.'

        if isinstance(self.modelcube, marvin.tools.modelcube.ModelCube):
            assert self.modelcube.is_binned() is False, 'a Spaxel cannot be binned.'

    def __repr__(self):
        """Spaxel representation."""

        if not self.loaded:
            return '<Marvin Spaxel (x={0.x:d}, y={0.y:d}, loaded=False)'.format(self)

        # Gets the coordinates relative to the centre of the cube/maps.
        y_mid, x_mid = np.array(self._parent_shape) / 2.
        x_centre = int(self.x - x_mid)
        y_centre = int(self.y - y_mid)

        return ('<Marvin Spaxel (plateifu={0.plateifu}, x={0.x:d}, y={0.y:d}; '
                'x_cen={1:d}, y_cen={2:d})>'.format(self, x_centre, y_centre))


class Bin(SpaxelBase):
    """A class that represents a bin."""

    def __init__(self, *args, **kwargs):

        super(Bin, self).__init__(*args, **kwargs)

        if isinstance(self.maps, marvin.tools.maps.Maps):
            assert self.maps.is_binned() is True, 'a Spaxel cannot be unbinned.'

        if isinstance(self.modelcube, marvin.tools.modelcube.ModelCube):
            assert self.modelcube.is_binned() is True, 'a Spaxel cannot be unbinned.'

        assert (isinstance(self.maps, marvin.tools.maps.Maps) or
                isinstance(self.modelcube, marvin.tools.modelcube.ModelCube)), \
            'a Bin must have a Maps or a ModelCube.'

        self.binid = None
        self.spaxels = None

    def load(self):
        """Loads quantities and spaxels."""

        super(Bin, self).load()
        self._create_spaxels()

    def __repr__(self):
        """Spaxel representation."""

        if not self.loaded:
            return '<Marvin Bin (x={0.x:d}, y={0.y:d}, loaded=False)'.format(self)

        # Gets the coordinates relative to the centre of the cube/maps.
        y_mid, x_mid = np.array(self._parent_shape) / 2.
        x_centre = int(self.x - x_mid)
        y_centre = int(self.y - y_mid)

        return ('<Marvin Bin (plateifu={0.plateifu}, x={0.x:d}, y={0.y:d}; '
                'x_cen={1:d}, y_cen={2:d})>'.format(self, x_centre, y_centre))

    def _create_spaxels(self):
        """Returns a list of the unbinned spaxels associated with this bin."""

        if self.maps:
            binid_map = self.maps.get_binid()
        elif self.modelcube:
            binid_map = self.modelcube.get_binid()

        # TODO: this may be an overkill and an extra API call. Remove it?
        if self.maps and self.modelcube:
            assert np.all(self.maps.get_binid() == self.modelcube.get_binid()), \
                'inconsistent binid arrays between Maps and ModelCube.'

        self.binid = binid_map[self.y, self.x]

        if self.binid < 0:
            raise MarvinError('coordinates ({}, {}) do not correspond to a valid binid.'
                              .format(self.x, self.y))

        spaxel_coords = zip(*np.where(binid_map == self.binid))
        self.spaxels = []

        for jj, ii in spaxel_coords:
            self.spaxels.append(Spaxel(x=jj, y=ii, plateifu=self.plateifu, release=self.release,
                                       cube=self.cube, maps=True if self.maps else False,
                                       modelcube=True if self.modelcube else False,
                                       bintype=None, template=self.template, lazy=True))

    def load_all(self):
        """Loads all the spaxels."""

        for spaxel in self.spaxels:
            spaxel.load()
