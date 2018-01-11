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
import inspect
import itertools
import six
import warnings

import numpy as np

import marvin
import marvin.core.core
import marvin.core.exceptions
from marvin.core.exceptions import MarvinError, MarvinUserWarning, MarvinBreadCrumb
import marvin.core.marvin_pickle
import marvin.utils.general.general

import marvin.tools.cube
import marvin.tools.maps
import marvin.tools.modelcube

from marvin.utils.general.structs import FuzzyDict
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

    Maps = marvin.tools.maps.Maps
    ModelCube = marvin.tools.modelcube.ModelCube

    if cls is not SpaxelBase:
        return type.__call__(cls, *args, **kwargs)

    cube = kwargs.pop('cube', None)
    maps = kwargs.pop('maps', None)
    modelcube = kwargs.pop('modelcube', None)

    plateifu = kwargs.pop('plateifu', (getattr(cube, 'plateifu', None) or
                                       getattr(maps, 'plateifu', None) or
                                       getattr(modelcube, 'plateifu', None)))

    mangaid = kwargs.pop('mangaid', (getattr(cube, 'mangaid', None) or
                                     getattr(maps, 'mangaid', None) or
                                     getattr(modelcube, 'mangaid', None)))

    release = kwargs.pop('release', (getattr(cube, 'release', None) or
                                     getattr(maps, 'release', None) or
                                     getattr(modelcube, 'release', None)))

    spaxel_kwargs = kwargs.copy()
    spaxel_kwargs.update(cube=cube, maps=maps, modelcube=modelcube,
                         plateifu=plateifu, mangaid=mangaid, release=release)

    if not cube and not maps and not modelcube:
        raise MarvinError('no inputs defined.')

    if not maps and not modelcube:
        return Spaxel(*args, **spaxel_kwargs)

    if isinstance(maps, Maps) or isinstance(modelcube, ModelCube):
        bintype = getattr(maps, 'bintype', None) or getattr(modelcube, 'bintype', None)
        spaxel_kwargs.update(bintype=bintype)
        if bintype.binned:
            return Bin(*args, **spaxel_kwargs)
        else:
            return Spaxel(*args, **spaxel_kwargs)

    if maps:
        maps = Maps((maps if maps is not True else None) or plateifu or mangaid,
                    release=release, **kwargs)
        spaxel_kwargs.update(bintype=maps.bintype)
        if maps.bintype.binned:
            return Bin(*args, **spaxel_kwargs)

    if modelcube:
        modelcube = ModelCube((modelcube if modelcube is not True else None) or
                              plateifu or mangaid, release=release, **kwargs)
        spaxel_kwargs.update(bintype=modelcube.bintype)
        if modelcube.bintype.binned:
            return Bin(*args, **spaxel_kwargs)

    return Spaxel(*args, **spaxel_kwargs)

    raise MarvinError('you have reached the end of the SpaxelBase logic. '
                      'This should never happen!')


# Overrides the __call__ method in abc.ABC.
SpaxelABC = abc.ABCMeta
SpaxelABC.__call__ = spaxel_factory


class DataModel(object):
    """A single object that holds the DRP and DAP datamodel."""

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
        mangaid (str):
            The mangaid of the cube/maps/modelcube of the spaxel to load.
        plateifu (str):
            The plate-ifu of the cube/maps/modelcube of the spaxel to load
            (either ``mangaid`` or ``plateifu`` can be used, but not both).
        cube (`~marvin.tools.cube.Cube` object or path or bool):
            If ``cube`` is a `~marvin.tools.cube.Cube` object, that
            cube will be used for the `.SpaxelBase` instantiation. This mode
            is mostly intended for `~marvin.utils.general.general.getSpaxel`
            as it significantly improves loading time. Otherwise, ``cube`` can
            be ``True`` (default), in which case a cube will be instantiated
            using the input ``filename``, ``mangaid``, or ``plateifu``. If
            ``cube=False``, no cube will be used and the cube associated
            quantities will not be available. ``cube`` can also be the
            path to the DRP cube to use.
        maps (`~marvin.tools.maps.Maps` object or path or bool)
            As ``cube`` but for the DAP measurements corresponding to the
            spaxel in the `.Maps`.
        modelcube (`marvin.tools.modelcube.ModelCube` object or path or bool)
            As ``maps`` but for the DAP measurements corresponding to the
            spaxel in the `.ModelCube`.
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

    def __init__(self, x, y, mangaid=None, plateifu=None,
                 cube=True, maps=True, modelcube=True, lazy=False, **kwargs):

        self.cube_quantities = FuzzyDict({})
        self.maps_quantities = FuzzyDict({})
        self.modelcube_quantities = FuzzyDict({})

        self._cube = cube
        self._maps = maps
        self._modelcube = modelcube

        self._plateifu = plateifu
        self._mangaid = mangaid

        self.kwargs = kwargs

        if not self._cube and not self._maps and not self._modelcube:
            raise MarvinError('either cube, maps, or modelcube must be True or '
                              'a Marvin Cube, Maps, or ModelCube object must be specified.')

        self._parent_shape = None

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinSpaxel {0}'.format(self.__class__),
                        category=self.__class__)

        self.x = int(x)
        self.y = int(y)

        self.loaded = False
        self.datamodel = None

        if lazy is False:
            self.load()

    def __dir__(self):

        class_members = list(list(zip(*inspect.getmembers(self.__class__)))[0])
        instance_attr = list(self.__dict__.keys())

        items = self.cube_quantities.__dir__()
        items += self.maps_quantities.__dir__()
        items += self.modelcube_quantities.__dir__()
        items += class_members + instance_attr

        return sorted(items)

    def __getattr__(self, value):

        _getattr = super(SpaxelBase, self).__getattribute__

        if value in _getattr('cube_quantities'):
            return _getattr('cube_quantities')[value]
        if value in _getattr('maps_quantities'):
            return _getattr('maps_quantities')[value]
        if value in _getattr('modelcube_quantities'):
            return _getattr('modelcube_quantities')[value]

        return super(SpaxelBase, self).__getattribute__(value)

    @abc.abstractmethod
    def __repr__(self):

        return '<SpaxelBase>'

    def _check_versions(self, attr):
        """Checks that all input object have the same versions.

        Runs sanity checks to make sure that ``attr`` has the same value
        in the input `.Cube`, `.Maps`, and `.ModelCube`. Returns the value
        for the attribute or ``None`` if the attribute does not exist.

        """

        out_value = None

        inputs = []
        for obj in [self._cube, self._maps, self._modelcube]:
            if obj is not None and not isinstance(obj, bool):
                inputs.append(obj)

        if len(inputs) == 1:
            return getattr(inputs[0], attr, None)

        for obj_a, obj_b in itertools.combinations(inputs, 2):
            if hasattr(obj_a, attr) and hasattr(obj_b, attr):
                assert getattr(obj_a, attr) == getattr(obj_b, attr), \
                    'inconsistent {!r} between {!r} and {!r}'.format(attr, obj_a, obj_b)
            out_value = getattr(obj_a, attr, None) or getattr(obj_b, attr, None)

        return out_value

    def _set_radec(self):
        """Calculates ra and dec for this spaxel."""

        self.ra = None
        self.dec = None

        for obj in [self._cube, self._maps, self._modelcube]:
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

        for attr in ['mangaid', 'plateifu', 'release', 'bintype', 'template']:
            self._check_versions(attr)

        self._plateifu = self.plateifu
        self._mangaid = self.mangaid

        self.datamodel = DataModel(self.release)

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

        if self._cube is False or self._cube is None:
            self._cube = None
            return
        elif not isinstance(self._cube, marvin.tools.cube.Cube):
            self._cube = self.getCube()

        self._parent_shape = self._cube._shape

        self.cube_quantities = self._cube._get_spaxel_quantities(self.x, self.y)

    def _load_maps(self):
        """Loads the cube and the properties."""

        if self._maps is False or self._maps is None:
            self._maps = None
            return
        elif not isinstance(self._maps, marvin.tools.maps.Maps):
            self._maps = self.getMaps()

        self._parent_shape = self._maps._shape

        self.maps_quantities = self._maps._get_spaxel_quantities(self.x, self.y)

    def _load_modelcube(self):
        """Loads the modelcube and associated arrays."""

        if self._modelcube is False or self._modelcube is None:
            self._modelcube = None
            return
        elif not isinstance(self._modelcube, marvin.tools.modelcube.ModelCube):

            if self.release == 'MPL-4':
                warnings.warn('ModelCube cannot be instantiated for MPL-4.', MarvinUserWarning)
                self._modelcube = None
                return

            self._modelcube = self.getModelCube()

        self._parent_shape = self._modelcube._shape

        self.modelcube_quantities = self._modelcube._get_spaxel_quantities(self.x, self.y)

    def getCube(self):
        """Returns the associated `~marvin.tools.cube.Cube`"""

        if isinstance(self._cube, marvin.tools.cube.Cube):
            return self._cube

        cube_kwargs = self.kwargs.copy()
        cube_kwargs.pop('bintype', None)
        cube_kwargs.pop('template', None)

        return marvin.tools.cube.Cube(
            (self._cube if self._cube is not True else None) or self._plateifu or self._mangaid,
            **cube_kwargs)

    def getMaps(self):
        """Returns the associated `~marvin.tools.maps.Maps`"""

        if isinstance(self._maps, marvin.tools.maps.Maps):
            return self._maps

        maps_kwargs = self.kwargs.copy()

        bintype = maps_kwargs.pop('bintype', None) or self.bintype
        template = maps_kwargs.pop('template', None) or self.template
        release = maps_kwargs.pop('release', None) or self.release

        return marvin.tools.maps.Maps(
            (self._maps if self._maps is not True else None) or self._plateifu or self._mangaid,
            bintype=bintype, template=template, release=release, **maps_kwargs)

    def getModelCube(self):
        """Returns the associated `~marvin.tools.modelcube.ModelCube`"""

        if isinstance(self._modelcube, marvin.tools.modelcube.ModelCube):
            return self._modelcube

        modelcube_kwargs = self.kwargs.copy()

        bintype = modelcube_kwargs.pop('bintype', None) or self.bintype
        template = modelcube_kwargs.pop('template', None) or self.template
        release = modelcube_kwargs.pop('release', None) or self.release

        return marvin.tools.modelcube.ModelCube(
            ((self._modelcube if self._modelcube is not True else None) or
             self._plateifu or self._mangaid),
            bintype=bintype, template=template, release=release, **modelcube_kwargs)

    @property
    def plateifu(self):
        """Returns the plateifu."""

        return self._check_versions('plateifu')

    @property
    def mangaid(self):
        """Returns the mangaid."""

        return self._check_versions('mangaid')

    @property
    def release(self):
        """Returns the release."""

        return self._check_versions('release')

    @property
    def bintype(self):
        """Returns the bintype."""

        return self._check_versions('bintype')

    @property
    def template(self):
        """Returns the template."""

        return self._check_versions('template')

    @property
    def manga_target1(self):
        """Return MANGA_TARGET1 flag."""
        return self.datamodel.drp.bitmasks['MANGA_TARGET1']

    @property
    def manga_target2(self):
        """Return MANGA_TARGET2 flag."""
        return self.datamodel.drp.bitmasks['MANGA_TARGET2']

    @property
    def manga_target3(self):
        """Return MANGA_TARGET3 flag."""
        return self.datamodel.drp.bitmasks['MANGA_TARGET3']

    @property
    def target_flags(self):
        """Bundle MaNGA targeting flags."""
        return [self.manga_target1, self.manga_target2, self.manga_target3]

    @property
    def quality_flags(self):
        """Bundle Cube DRP3QUAL and Maps DAPQUAL flags."""

        drp3qual = self.datamodel.drp.bitmasks['MANGA_DRP3QUAL']
        cube = self.getCube()
        drp3qual.mask = int(cube.header['DRP3QUAL'])
        qual_flags = [drp3qual]

        if self.release != 'MPL-4':
            qual_flags.append(self.datamodel.dap.bitmasks['MANGA_DAPQUAL'])

        return qual_flags


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

        if isinstance(self._maps, marvin.tools.maps.Maps):
            assert self._maps.is_binned() is False, 'a Spaxel cannot be binned.'

        if isinstance(self._modelcube, marvin.tools.modelcube.ModelCube):
            assert self._modelcube.is_binned() is False, 'a Spaxel cannot be binned.'

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

        self.binid = None
        self.spaxels = None

        super(Bin, self).__init__(*args, **kwargs)

        if isinstance(self._maps, marvin.tools.maps.Maps):
            assert self._maps.is_binned() is True, 'a Spaxel cannot be unbinned.'

        if isinstance(self._modelcube, marvin.tools.modelcube.ModelCube):
            assert self._modelcube.is_binned() is True, 'a Spaxel cannot be unbinned.'

        assert (isinstance(self._maps, marvin.tools.maps.Maps) or
                isinstance(self._modelcube, marvin.tools.modelcube.ModelCube)), \
            'a Bin must have a Maps or a ModelCube.'

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
                'x_cen={1:d}, y_cen={2:d}, n_spaxels={3})>'.format(self, x_centre, y_centre, len(self.spaxels)))

    def _create_spaxels(self):
        """Returns a list of the unbinned spaxels associated with this bin."""

        if self._maps:
            binid_map = self._maps.get_binid()
        elif self._modelcube:
            binid_map = self._modelcube.get_binid()

        # TODO: this may be an overkill and an extra API call. Remove it?
        if self._maps and self._modelcube:
            assert np.all(self._maps.get_binid() == self._modelcube.get_binid()), \
                'inconsistent binid arrays between Maps and ModelCube.'

        self.binid = binid_map[self.y, self.x]

        if self.binid < 0:
            raise MarvinError('coordinates ({}, {}) do not correspond to a valid binid.'
                              .format(self.x, self.y))

        spaxel_coords = zip(*np.where(binid_map == self.binid))
        self.spaxels = []

        for jj, ii in spaxel_coords:
            self.spaxels.append(Spaxel(x=jj, y=ii, plateifu=self.plateifu, release=self.release,
                                       cube=self._cube, maps=True if self._maps else False,
                                       modelcube=True if self._modelcube else False,
                                       bintype=None, template=self.template, lazy=True))

    def load_all(self):
        """Loads all the spaxels."""

        for spaxel in self.spaxels:
            spaxel.load()
