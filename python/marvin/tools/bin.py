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

from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.tools.maps import Maps, _is_MPL4
from marvin.tools.modelcube import ModelCube
from marvin.tools.spaxel import Spaxel


class Bin(object):

    def __init__(self, binid, **kwargs):

        self._maps, self._modelcube = self._get_dap_objects(**kwargs)
        self.binid = binid

        spaxel_coords = self._maps.get_bin_spaxels(binid, only_list=True)

        if len(spaxel_coords) == 0:
            self.spaxels = []
        else:
            self.spaxels = [Spaxel(x=cc[0], y=cc[1], cube=True, maps=self._maps,
                                   modelcube=self._modelcube, load=False, **kwargs)
                            for cc in spaxel_coords]

    def _get_dap_objects(self, **kwargs):
        """Gets the Maps and ModelCube object."""

        try:
            maps = Maps(**kwargs)
        except MarvinError as ee:
            raise MarvinError('failed to open a Maps: {0}'.format(str(ee)))

        if _is_MPL4(maps._dapver):
            return maps, None

        try:
            modelcube = ModelCube(**kwargs)
        except MarvinError:
            warnings.warn('cannot open a ModelCube for this combination of '
                          'parameters. Some fetures will not be available.', MarvinUserWarning)
            modelcube = False

        return maps, modelcube
