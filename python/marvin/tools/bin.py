#!/usr/bin/env python
# encoding: utf-8
#
# bin.py
#
# Created by José Sánchez-Gallego on 6 Nov 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import warnings

from marvin.core.exceptions import MarvinError, MarvinUserWarning, MarvinBreadCrumb
from marvin.tools.maps import Maps, _is_MPL4
from marvin.tools.modelcube import ModelCube
from marvin.tools.spaxel import Spaxel

breadcrumb = MarvinBreadCrumb()


class Bin(object):
    """A class to interface with a bin.

    This class represents a bin of spaxels in a MaNGA DAP
    :class:`~marvin.tools.maps.Maps` or
    :class:`~marvin.tools.modelcube.ModelCube`.
    By definition, a bin is identified by a ``binid`` and can correspond to a
    binned or unbinned MAPS, although in the latter case there is no formal
    difference between a bin and a spaxel.

    When a bin is instantiated, a list of its contituent
    :class:`~marvin.tools.spaxel.Spaxel` is created. ``Spaxel`` objects are
    created unloaded for efficiency, but they can be loaded by doing, for
    example for the first spaxel, ``bin.spaxels[0].load()``.

    MAPS properties and (if available) Model Cube spectra are loaded for the
    bin.

    Parameters:
        binid (int):
            The binid of the requested bin.
        extra_kwargs:
            Any other keyword parameter necessary to instantiate the
            :class:`~marvin.tools.maps.Maps`, the
            :class:`~marvin.tools.modelcube.ModelCube`, or the
            :class:`~marvin.tools.spaxel.Spaxel` related to this bin. Refer to
            the documentation of those classes for details on the available
            keywords.

    Attributes:
        spaxels (list):
            A list of :class:`~marvin.tools.spaxel.Spaxel` objects that form
            the bin. The spaxels are created unloaded.
        properties (dict):
            Same as the :class:`~marvin.tools.maps.Maps` properties for a
            given spaxel, but in this case for the bin.
        modelcube_attributes:
            All the attributes derived for a given spaxel from the
            :class:`~marvin.tools.modelcube.ModelCube`, but in this case for
            the bin. If a modelcube is not available or it is not instantiated,
            these attributes will be ``None``.

    """

    def __init__(self, binid, **kwargs):

        self.plateifu = None
        self.mangaid = None

        self._maps, self._modelcube = self._get_dap_objects(**kwargs)
        self.binid = binid

        # Drops some keyword that could make spaxel load fail.
        kwargs.pop('bintype', None)
        kwargs.pop('mode', None)

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinBin {0}'.format(self.__class__),
                        category=self.__class__)

        self._load_spaxels(**kwargs)
        self._load_data(**kwargs)

        self.release = self._maps.release

    def __repr__(self):
        return ('<Marvin Bin (binid={0}, n_spaxels={1}, bintype={2}, template_kin={3})>'
                .format(self.binid, len(self.spaxels), self._maps.bintype,
                        self._maps.template_kin))

    def _get_dap_objects(self, **kwargs):
        """Gets the Maps and ModelCube object."""

        try:
            kwargs_maps = kwargs.copy()
            kwargs_maps.pop('modelcube_filename', None)
            kwargs_maps['filename'] = kwargs_maps.pop('maps_filename', None)
            maps = Maps(**kwargs_maps)
        except MarvinError as ee:
            raise MarvinError('failed to open a Maps: {0}'.format(str(ee)))

        self.plateifu = maps.plateifu
        self.mangaid = maps.mangaid

        if _is_MPL4(maps._dapver):
            return maps, None

        # try:
        kwargs_modelcube = kwargs.copy()
        kwargs_modelcube.pop('maps_filename', None)
        kwargs_modelcube['filename'] = kwargs_modelcube.pop('modelcube_filename', None)
        kwargs_modelcube['plateifu'] = None

        # TODO: we need a check here to make sure that if we open both Maps and ModelCube
        # from files, their bintypes, templates, plate-ifu, etc are consistent.

        # TODO: this is not a very good implementation and probably has corner cases in which
        # it fails. It should be refactored.

        if kwargs_modelcube['filename'] is None:
            kwargs_modelcube['plateifu'] = self.plateifu
            kwargs_modelcube['bintype'] = maps.bintype
            kwargs_modelcube['template_kin'] = maps.template_kin
            kwargs_modelcube['template_pop'] = maps.template_pop
            kwargs_modelcube['release'] = maps._release

        modelcube = ModelCube(**kwargs_modelcube)

        # except Exception:
        #
        #     warnings.warn('cannot open a ModelCube for this combination of '
        #                   'parameters. Some fetures will not be available.', MarvinUserWarning)
        #     modelcube = False

        return maps, modelcube

    def _load_spaxels(self, **kwargs):
        """Creates a list of unloaded spaxels for this binid."""

        load_spaxels = kwargs.pop('load_spaxels', False)

        self.spaxel_coords = self._maps.get_bin_spaxels(self.binid, only_list=True)

        if len(self.spaxel_coords) == 0:
            raise MarvinError('there are no spaxels associated with binid={0}.'.format(self.binid))
        else:

            if 'plateifu' not in kwargs:
                kwargs['plateifu'] = self._maps.plateifu

            modelcube_for_spaxel = False if not self._modelcube else self._modelcube.get_unbinned()
            self.spaxels = [Spaxel(x=cc[0], y=cc[1], cube=True, maps=self._maps.get_unbinned(),
                                   modelcube=modelcube_for_spaxel,
                                   load=load_spaxels,
                                   **kwargs)
                            for cc in self.spaxel_coords]

    def _load_data(self, **kwargs):
        """Loads one of the spaxels to get the DAP properties for the binid."""

        assert len(self.spaxel_coords) > 0

        sample_coords = self.spaxel_coords[0]

        if 'plateifu' not in kwargs:
            kwargs['plateifu'] = self._maps.plateifu

        sample_spaxel = Spaxel(x=sample_coords[0], y=sample_coords[1],
                               cube=True, maps=self._maps, modelcube=self._modelcube,
                               load=True, allow_binned=True, **kwargs)

        self.specres = sample_spaxel.specres
        self.specresd = sample_spaxel.specresd
        self.spectrum = sample_spaxel.spectrum
        self.properties = sample_spaxel.properties

        self.model_flux = sample_spaxel.model_flux
        self.redcorr = sample_spaxel.redcorr
        self.model = sample_spaxel.model
        self.emline = sample_spaxel.emline
        self.emline_base = sample_spaxel.emline_base
        self.stellar_continuum = sample_spaxel.stellar_continuum

    def load_all(self):
        """Loads all the spaxels that for this bin."""

        for spaxel in self.spaxels:
            spaxel.load()

        return
