#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-11-03
# @Filename: spaxel.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   andrews
# @Last modified time: 2018-10-16 10:10:58


from __future__ import absolute_import, division, print_function

import inspect
import itertools
import warnings

import numpy as np
import six

import marvin
import marvin.core.exceptions
import marvin.core.marvin_pickle
import marvin.tools.cube
import marvin.tools.maps
import marvin.tools.modelcube
import marvin.utils.general.general
from marvin.core.exceptions import MarvinBreadCrumb, MarvinError, MarvinUserWarning
from marvin.utils.datamodel.dap import datamodel as dap_datamodel
from marvin.utils.datamodel.drp import datamodel as drp_datamodel
from marvin.utils.general.structs import FuzzyDict


breadcrumb = MarvinBreadCrumb()


class DataModel(object):
    """A single object that holds the DRP and DAP datamodel."""

    def __init__(self, release):

        self.drp = drp_datamodel[release]
        self.dap = dap_datamodel[release]


class Spaxel(object):
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
            An object containing the DRP and DAP datamodels.
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

    def __init__(self, x, y, cube=True, maps=True, modelcube=True, lazy=False, **kwargs):

        if not cube and not maps and not modelcube:
            raise MarvinError('no inputs defined.')

        self.cube_quantities = FuzzyDict({})
        self.maps_quantities = FuzzyDict({})
        self.modelcube_quantities = FuzzyDict({})

        self._cube = cube
        self._maps = maps
        self._modelcube = modelcube

        for attr in ['mangaid', 'plateifu', 'release', 'bintype', 'template']:

            value = kwargs.pop(attr, None) or \
                getattr(cube, attr, None) or \
                getattr(maps, attr, None) or \
                getattr(modelcube, attr, None)

            setattr(self, attr, value)

        self._kwargs = kwargs
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

        # Load VACs
        from marvin.contrib.vacs.base import VACMixIn
        self.vacs = VACMixIn.get_vacs(self)

    def __dir__(self):

        class_members = list(list(zip(*inspect.getmembers(self.__class__)))[0])
        instance_attr = list(self.__dict__.keys())

        items = self.cube_quantities.__dir__()
        items += self.maps_quantities.__dir__()
        items += self.modelcube_quantities.__dir__()
        items += class_members + instance_attr

        return sorted(items)

    def __getattr__(self, value):

        _getattr = super(Spaxel, self).__getattribute__

        for tool_quantity_dict in ['cube_quantities', 'maps_quantities', 'modelcube_quantities']:
            if value in _getattr(tool_quantity_dict):
                return _getattr(tool_quantity_dict)[value]

        return super(Spaxel, self).__getattribute__(value)

    def __repr__(self):
        """Spaxel representation."""

        if not self.loaded:
            return '<Marvin Spaxel (x={0.x:d}, y={0.y:d}, loaded=False)'.format(self)

        # Gets the coordinates relative to the centre of the cube/maps.
        y_mid, x_mid = np.array(self._parent_shape) / 2.
        x_centre = int(self.x - x_mid)
        y_centre = int(self.y - y_mid)

        # Determine what tools are loaded.
        tools = np.array(['cube', 'maps', 'modelcube'])
        load_idx = np.where([self._cube, self._maps, self._modelcube])[0]
        flags = '/'.join(tools[load_idx])

        return ('<Marvin Spaxel (plateifu={0.plateifu}, x={0.x:d}, y={0.y:d}; '
                'x_cen={1:d}, y_cen={2:d}, loaded={3})>'.format(self, x_centre, y_centre, flags))

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

    def save(self, path, overwrite=False):
        """Pickles the spaxel to a file.

        Parameters:
            path (str):
                The path of the file to which the `.Spaxel` will be saved.
                Unlike for other Marvin Tools that derive from
                `~marvin.tools.core.MarvinToolsClass`, ``path`` is
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

    def load(self, force=None):
        """Loads the spaxel data.

        Loads the spaxel data for cubes/maps/modelcube. By default attempts
        to load whatever is specified when spaxels are instantianted from other
        Marvin Tools. Can manually force load a data type with the force
        keyword.

        Parameters:
        -----------
        force : {cube|maps|modelcube}
            Datatype to force load.

        """

        if self.loaded and force is None:
            warnings.warn('already loaded', MarvinUserWarning)
            return

        assert force in [None, 'cube', 'maps', 'modelcube'], \
            'force can only be cube, maps, or modelcube'

        for tool in ['cube', 'maps', 'modelcube']:
            self._load_tool(tool, force=(force is not None and force == tool))

        self._set_radec()
        self.loaded = True

        for attr in ['mangaid', 'plateifu', 'release', 'bintype', 'template']:
            setattr(self, attr, self._check_versions(attr))

        self.datamodel = DataModel(self.release)

    def _load_tool(self, tool, force=False):
        """Loads the tool and the associated quantities."""

        if tool == 'cube':
            class_name = marvin.tools.cube.Cube
            method = self.getCube
            quantities_dict = 'cube_quantities'
        elif tool == 'maps':
            class_name = marvin.tools.maps.Maps
            method = self.getMaps
            quantities_dict = 'maps_quantities'
        elif tool == 'modelcube':
            class_name = marvin.tools.modelcube.ModelCube
            method = self.getModelCube
            quantities_dict = 'modelcube_quantities'

        attr_value = getattr(self, '_' + tool)
        if (attr_value is False or attr_value is None) and force is False:
            setattr(self, '_' + tool, None)
            return

        if not isinstance(attr_value, class_name):
            if tool == 'modelcube' and self.release == 'MPL-4':
                warnings.warn('ModelCube cannot be instantiated for MPL-4.', MarvinUserWarning)
                self._modelcube = None
                return
            else:
                setattr(self, '_' + tool, method())
        else:
            if force is True:
                warnings.warn('{0} is already loaded'.format(tool), MarvinUserWarning)

        self._parent_shape = getattr(getattr(self, '_' + tool), '_shape')

        setattr(self, quantities_dict,
                getattr(getattr(self, '_' + tool), '_get_spaxel_quantities')(self.x, self.y,
                                                                             spaxel=self))

    def getCube(self):
        """Returns the associated `~marvin.tools.cube.Cube`"""

        if isinstance(self._cube, marvin.tools.cube.Cube):
            return self._cube

        cube_input = (self._cube if self._cube is not True else None) \
            or self.plateifu or self.mangaid

        return marvin.tools.cube.Cube(cube_input, release=self.release, **self._kwargs)

    def getMaps(self):
        """Returns the associated `~marvin.tools.maps.Maps`"""

        if isinstance(self._maps, marvin.tools.maps.Maps):
            return self._maps

        maps_input = (self._maps if self._maps is not True else None) \
            or self.plateifu or self.mangaid

        return marvin.tools.maps.Maps(maps_input, bintype=self.bintype,
                                      template=self.template, release=self.release,
                                      **self._kwargs)

    def getModelCube(self):
        """Returns the associated `~marvin.tools.modelcube.ModelCube`"""

        if isinstance(self._modelcube, marvin.tools.modelcube.ModelCube):
            return self._modelcube

        modelcube_input = (self._modelcube if self._modelcube is not True else None) \
            or self.plateifu or self.mangaid

        return marvin.tools.modelcube.ModelCube(modelcube_input, bintype=self.bintype,
                                                template=self.template, release=self.release,
                                                **self._kwargs)

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
