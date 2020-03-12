#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-13
# @Filename: dapall.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-14 12:03:58

import distutils
import os

import astropy.io.fits

import marvin
from marvin.core.exceptions import MarvinError
from marvin.utils.general import get_dapall_path, map_dapall


__all__ = ['DAPallMixIn']


class DAPallMixIn(object):
    """A mixin that provides access to DAPall paremeters.

    Must be used in combination with `.MarvinToolsClass` and initialised
    before `~.DAPallMixIn.dapall` can be called.

    `DAPallMixIn` uses the `.MarvinToolsClass.data_origin` of the object to
    determine how to obtain the DAPall information. However, if the object
    contains a ``dapall`` attribute with the path to a DAPall file, that file
    will be used.

    """

    __min_dapall_version__ = distutils.version.StrictVersion('2.1.0')

    @property
    def dapall(self):
        """Returns the contents of the DAPall data for this target."""

        if (not self._dapver or
                distutils.version.StrictVersion(self._dapver) < self.__min_dapall_version__):
            raise MarvinError('DAPall is not available for versions before MPL-6.')

        if hasattr(self, '_dapall') and self._dapall is not None:
            return self._dapall

        if self.data_origin == 'file':
            try:
                dapall_data = self._get_dapall_from_file()
            except IOError:
                marvin.log.debug('cannot find DAPall file. Trying remote request.')
                dapall_data = self._get_from_api()
        elif self.data_origin == 'db':
            dapall_data = self._get_dapall_from_db()
        else:
            dapall_data = self._get_dapall_from_api()

        self._dapall = dapall_data

        return self._dapall

    def _get_dapall_from_file(self):
        """Uses DAPAll file to retrieve information."""

        daptype = self.bintype.name + '-' + self.template.name

        dapall_path = get_dapall_path(self._drpver, self._dapver)

        assert dapall_path is not None, 'cannot build DAPall file.'

        if not os.path.exists(dapall_path):
            raise MarvinError('cannot find DAPall file in the system.')

        dapall_hdu = astropy.io.fits.open(dapall_path)

        header = dapall_hdu[0].header
        dapall_table = dapall_hdu[-1].data

        dapall_row = dapall_table[(dapall_table['PLATEIFU'] == self.plateifu) &
                                  (dapall_table['DAPTYPE'] == daptype)]

        assert len(dapall_row) == 1, 'cannot find matching row in DAPall.'

        return map_dapall(header, dapall_row[0])

    def _get_dapall_from_db(self):
        """Uses the DB to retrieve the DAPAll data."""

        dapall_data = {}

        daptype = self.bintype.name + '-' + self.template.name

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise MarvinError('No DB connected')

        datadb = mdb.datadb
        dapdb = mdb.dapdb

        dapall_row = mdb.session.query(dapdb.DapAll).join(
            dapdb.File, datadb.PipelineInfo, datadb.PipelineVersion).filter(
                mdb.datadb.PipelineVersion.version == self._dapver,
                dapdb.DapAll.plateifu == self.plateifu,
                dapdb.DapAll.daptype == daptype).use_cache().first()

        if dapall_row is None:
            raise MarvinError('cannot find a DAPall match for this target in the DB.')

        for col in dapall_row.__table__.columns.keys():
            if col != 'pk' and '_pk' not in col:
                dapall_data[col] = getattr(dapall_row, col)

        return dapall_data

    def _get_dapall_from_api(self):
        """Uses the API to retrieve the DAPall data."""

        url = marvin.config.urlmap['api']['dapall']['url']

        url_full = url.format(name=self.plateifu,
                              bintype=self.bintype.name,
                              template=self.template.name)

        try:
            response = self._toolInteraction(url_full)
        except Exception as ee:
            raise MarvinError('found a problem while getting DAPall: {0}'.format(str(ee)))

        if response.results['error'] is not None:
            raise MarvinError('found a problem while getting DAPall: {}'
                              .format(str(response.results['error'])))

        data = response.getData()

        return data['dapall_data']
