#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-13
# @Filename: nsa.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego
# @Last modified time: 2018-07-13 16:54:57

import warnings

from brain.core.exceptions import BrainError

from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils.general import get_nsa_data


__all__ = ['NSAMixIn']


class NSAMixIn(object):
    """A mixin that provides access to NSA paremeters.

    Must be used in combination with `.MarvinToolsClass` and initialised
    before `~.NSAMixIn.nsa` can be called.

    Parameters:
        nsa_source ({'auto', 'drpall', 'nsa'}):
            Defines how the NSA data for this object should loaded when
            ``.nsa`` is first called. If ``drpall``, the drpall file will
            be used (note that this will only contain a subset of all the NSA
            information); if ``nsa``, the full set of data from the DB will be
            retrieved. If the drpall file or a database are not available, a
            remote API call will be attempted. If ``nsa_source='auto'``, the
            source will depend on how the parent object has been
            instantiated. If the parent has ``data_origin='file'``,
            the drpall file will be used (as it is more likely that the user
            has that file in their system). Otherwise, ``nsa_source='nsa'``
            will be assumed. This behaviour can be modified during runtime by
            modifying the ``nsa_mode`` attribute with one of the valid values.

    """

    def __init__(self, nsa_source='auto'):

        self._nsa = None
        self.nsa_source = nsa_source

        assert self.nsa_source in ['auto', 'nsa', 'drpall'], \
            'nsa_source must be one of auto, nsa, or drpall'

    @property
    def nsa(self):
        """Returns the contents of the NSA catalogue for this target."""

        if hasattr(self, 'nsa_source') and self.nsa_source is not None:
            nsa_source = self.nsa_source
        else:
            nsa_source = 'auto'

        if self._nsa is None:

            if nsa_source == 'auto':
                if self.data_origin == 'file':
                    nsa_source = 'drpall'
                else:
                    nsa_source = 'nsa'

            try:
                self._nsa = get_nsa_data(self.mangaid, mode='auto',
                                         source=nsa_source,
                                         drpver=self._drpver,
                                         drpall=self._drpall)
            except (MarvinError, BrainError):
                warnings.warn('cannot load NSA information for mangaid={!r}.'
                              .format(self.mangaid), MarvinUserWarning)
                return None

        return self._nsa
