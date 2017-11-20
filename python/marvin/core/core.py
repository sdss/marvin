#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Filename: core.py
# @License: BSD 3-Clause
# @Copyright: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import abc
import distutils
import os
import re
import six
import warnings

import astropy.io.fits

from brain.core.exceptions import BrainError

import marvin
import marvin.api.api

from marvin.core import marvin_pickle
from marvin.core.exceptions import MarvinUserWarning, MarvinError
from marvin.core.exceptions import MarvinMissingDependency, MarvinBreadCrumb

from marvin.utils.db import testDbConnection
from marvin.utils.general import mangaid2plateifu, get_nsa_data, get_dapall_file, map_dapall

try:
    from sdss_access.path import Path
except ImportError:
    Path = None

try:
    from sdss_access import RsyncAccess
except ImportError:
    RsyncAccess = None


__ALL__ = ['MarvinToolsClass']


def kwargsGet(kwargs, key, replacement):
    """As kwargs.get but uses replacement if the value is None."""

    if key not in kwargs:
        return replacement
    elif key in kwargs and kwargs[key] is None:
        return replacement
    else:
        return kwargs[key]


breadcrumb = MarvinBreadCrumb()


class MarvinToolsClass(object, six.with_metaclass(abc.ABCMeta)):
    """Marvin tools main base class.

    This super class implements the :ref:`decision tree <marvin-dma>`
    for using local files, database, or remote connection when
    initialising a Marvin tools object.

    Parameters:
        input (str):
            A string that can be a filename, plate-ifu, or mangaid. It will be
            aumatically identified based on its unique format. This argument
            is always the first one, so it can be defined without the keyword
            for convenience.
        filename (str):
            The path of the file containing the file to load. If set,
            ``input`` is ignored.
        mangaid (str):
            The mangaid of the file to load. If set, ``input`` is ignored.
        plateifu (str):
            The plate-ifu of the data cube to load. If set, ``input`` is
            ignored.
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :ref:`mode-decision-tree`.
        data (:class:`~astropy.io.fits.HDUList`, SQLAlchemy object, or None):
            An astropy ``HDUList`` or a SQLAlchemy object, to be used for
            initialisation. If ``None``, the :ref:`normal <marvin-dma>`` mode
            will be used.
        release (str):
            The MPL/DR version of the data to use.
        drpall (str):
            The path to the
            `drpall <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/metadata#DRP:DRPall>`_
            file to use. If not set it will use the default path for the file
            based on the ``release``
        download (bool):
            If ``True``, the data will be downloaded on instantiation. See
            :ref:`marvin-download-objects`.

    Attributes:
        data (:class:`~astropy.io.fits.HDUList`, SQLAlchemy object, or dict):
            Depending on the access mode, ``data`` is populated with the
            |HDUList| from the FITS file, a
            `SQLAlchemy <http://www.sqlalchemy.org>`_ object, or a dictionary
            of values returned by an API call.
        datamodel:
            A datamodel object, whose type depends on the subclass that
            initialises the datamodel.
        data_origin ({'file', 'db', 'api'}):
            Indicates the origin of the data, either from a file, the DB, or
            an API call.
        filename (str):
            The path of the file used, if any.
        mangaid (str):
            The mangaid of the target.
        plateifu:
            The plateifu of the target

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None, drpall=None, download=None):

        self.data = data
        self.data_origin = None

        self.filename = filename
        self.mangaid = mangaid
        self.plateifu = plateifu

        self.mode = mode if mode is not None else marvin.config.mode

        self._release = release if release is not None else marvin.config.release

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)
        self._drpall = marvin.config._getDrpAllPath(self._drpver) if drpall is None else drpall

        self._forcedownload = download if download is not None else marvin.config.download

        # Sets filename, plateifu, and mangaid depending on the values the input parameters.
        self._determine_inputs(input)

        self.datamodel = None
        self._set_datamodel()

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinTool {0}'.format(self.__class__),
                        category=self.__class__)

        assert self.mode in ['auto', 'local', 'remote']
        assert self.filename is not None or self.plateifu is not None, 'no inputs set.'

        if self.mode == 'local':
            self._doLocal()
        elif self.mode == 'remote':
            self._doRemote()
        elif self.mode == 'auto':
            try:
                self._doLocal()
            except Exception as ee:
                if self.filename:
                    # If the input contains a filename we don't want to go into remote mode.
                    raise(ee)
                else:
                    warnings.warn('local mode failed. Trying remote now.', MarvinUserWarning)
                    self._doRemote()

        # Sanity check to make sure data_origin has been properly set.
        assert self.data_origin in ['file', 'db', 'api'], 'data_origin is not properly set.'

    def _determine_inputs(self, input):
        """Determines what inputs to use in the decision tree."""

        if input:

            assert self.filename is None and self.plateifu is None and self.mangaid is None, \
                'if input is set, filename, plateifu, and mangaid cannot be set.'

            assert isinstance(input, six.string_types), 'input must be a string.'

            input_dict = self._parse_input(input)

            if input_dict['plate'] is not None and input_dict['ifu'] is not None:
                self.plateifu = input
            elif input_dict['plate'] is not None and input_dict['ifu'] is None:
                self._plate = input
            elif input_dict['mangaid'] is not None:
                self.mangaid = input
            else:
                # Assumes the input must be a filename
                self.filename = input

        if self.filename is None and self.mangaid is None and self.plateifu is None:
            raise MarvinError('no inputs defined.')

        if self.filename:
            self.mangaid = None
            self.plateifu = None

            if self.mode == 'remote':
                raise MarvinError('filename not allowed in remote mode.')

            assert os.path.exists(self.filename), \
                'filename {} does not exist.'.format(str(self.filename))

        elif self.plateifu:
            assert not self.filename, 'invalid set of inputs.'

        elif self.mangaid:
            assert not self.filename, 'invalid set of inputs.'
            self.plateifu = mangaid2plateifu(self.mangaid,
                                             drpall=self._drpall,
                                             drpver=self._drpver)

        elif self._plate:
            assert not self.filename, 'invalid set of inputs.'

    @staticmethod
    def _parse_input(value):
        """Parses and input and determines plate, ifu, and mangaid."""

        # Number of IFUs per size
        n_ifus = {19: 2, 37: 4, 61: 4, 91: 2, 127: 5, 7: 12}

        return_dict = {'plate': None, 'ifu': None, 'mangaid': None}

        plateifu_pattern = re.compile('([0-9]{4,5})\-([0-9]{4,9})')
        ifu_pattern = re.compile('(7|127|[0-9]{2})([0-9]{2})')
        mangaid_pattern = re.compile('[0-9]{1,3}\-[0-9]+')
        plateid_pattern = re.compile('([0-9]{4,})(?!-)(?<!-)')

        plateid_match = re.match(plateid_pattern, value)
        plateifu_match = re.match(plateifu_pattern, value)
        mangaid_match = re.match(mangaid_pattern, value)

        # Check whether the input value matches the plateifu pattern
        if plateifu_match is not None:
            plate, ifu = plateifu_match.groups(0)

            # If the value matches a plateifu, checks that the ifu is a valid one.
            ifu_match = re.match(ifu_pattern, ifu)
            if ifu_match is not None:
                ifu_size, ifu_id = map(int, ifu_match.groups(0))
                if ifu_id <= n_ifus[ifu_size]:
                    return_dict['plate'] = plate
                    return_dict['ifu'] = ifu

        # Check whether this is a mangaid
        elif mangaid_match is not None:
            return_dict['mangaid'] = value

        # Check whether this is a plate
        elif plateid_match is not None:
            return_dict['plate'] = value

        return return_dict

    def _doLocal(self):
        """Tests if it's possible to load the data locally."""

        if self.filename:

            if os.path.exists(self.filename):
                self.mode = 'local'
                self.data_origin = 'file'
            else:
                raise MarvinError('input file {0} not found'.format(self.filename))

        elif self.plateifu:

            testDbConnection(marvin.marvindb.session)

            if marvin.marvindb.db:
                self.mode = 'local'
                self.data_origin = 'db'
            else:
                fullpath = self._getFullPath()

                if fullpath and os.path.exists(fullpath):
                    self.mode = 'local'
                    self.filename = fullpath
                    self.data_origin = 'file'
                else:
                    if self._forcedownload:
                        self.download()
                        self.data_origin = 'file'
                    else:
                        raise MarvinError('this is the end of the road. '
                                          'Try using some reasonable inputs.')

    def _doRemote(self):
        """Tests if remote connection is possible."""

        if self.filename:
            raise MarvinError('filename not allowed in remote mode.')
        else:
            self.mode = 'remote'
            self.data_origin = 'api'

    @abc.abstractmethod
    def _set_datamodel(self):
        """Sets the datamodel for this object. Must be overridden by each subclass."""

        pass

    def download(self, pathType=None, **pathParams):
        """Download using sdss_access Rsync"""

        if not RsyncAccess:
            raise MarvinError('sdss_access is not installed')
        else:
            rsync_access = RsyncAccess()
            rsync_access.remote()
            rsync_access.add(pathType, **pathParams)
            rsync_access.set_stream()
            rsync_access.commit()
            paths = rsync_access.get_paths()
            self.filename = paths[0]  # doing this for single files, may need to change

    @abc.abstractmethod
    def _getFullPath(self, pathType=None, url=None, **pathParams):
        """Returns the full path of the file in the tree.

        This method must be overridden by each subclass.

        """

        if not Path:
            raise MarvinMissingDependency('sdss_access is not installed')
        else:
            try:
                if url:
                    fullpath = Path().url(pathType, **pathParams)
                else:
                    fullpath = Path().full(pathType, **pathParams)
            except Exception as ee:
                warnings.warn('sdss_access was not able to retrieve the full path of the file. '
                              'Error message is: {0}'.format(str(ee)), MarvinUserWarning)
                fullpath = None

        return fullpath

    def _toolInteraction(self, url, params=None):
        """Runs an Interaction and passes self._release."""

        params = params or {'release': self._release}
        return marvin.api.api.Interaction(url, params=params)

    def __getstate__(self):

        if self.data_origin == 'db':
            raise MarvinError('objects with data_origin=\'db\' cannot be saved.')

        odict = self.__dict__.copy()
        del odict['data']

        return odict

    def __setstate__(self, idict):

        data = None
        if idict['data_origin'] == 'file':
            try:
                data = astropy.io.fits.open(idict['filename'])
            except Exception as ee:
                warnings.warn('there was a problem reloading the FITS object: {0}. '
                              'The object has been unpickled but not all the functionality '
                              'will be available.'.format(str(ee)), MarvinUserWarning)

        self.__dict__.update(idict)
        self.data = data

    def save(self, path=None, overwrite=False):
        """Pickles the object.

        If ``path=None``, uses the default location of the file in the tree
        but changes the extension of the file to ``.mpf``. Returns the path
        of the saved pickle file.

        Parameters:
            obj:
                Marvin object to pickle.
            path (str):
                Path of saved file. Default is ``None``.
            overwrite (bool):
                If ``True``, overwrite existing file. Default is ``False``.

        Returns:
            str:
                Path of saved file.

        """

        return marvin_pickle.save(self, path=path, overwrite=overwrite)

    @classmethod
    def restore(cls, path, delete=False):
        """Restores a MarvinToolsClass object from a pickled file.

        If ``delete=True``, the pickled file will be removed after it has been
        unplickled. Note that, for objects with ``data_origin='file'``, the
        original file must exists and be in the same path as when the object
        was first created.

        """

        return marvin_pickle.restore(path, delete=delete)

    @property
    def release(self):
        """Returns the release."""

        return self._release

    @release.setter
    def release(self, value):
        """Fails when trying to set the release after instatiation."""

        raise MarvinError('the release cannot be changed once the object has been instantiated.')

    @property
    def plate(self):
        """Returns the plate id."""

        return int(self.plateifu.split('-')[0])

    @property
    def ifu(self):
        """Returns the IFU."""

        return int(self.plateifu.split('-')[1])

    def __del__(self):
        """Destructor for closing FITS files."""

        if self.data_origin == 'file' and isinstance(self.data, astropy.io.fits.HDUList):
            try:
                self.data.close()
            except Exception as ee:
                warnings.warn('failed to close FITS instance: {0}'.format(ee), MarvinUserWarning)


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
                warnings.warn('cannot find DAPall file. Trying remote request.',
                              MarvinUserWarning)
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

        dapall_path = get_dapall_file(self._drpver, self._dapver)

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
                dapdb.DapAll.daptype == daptype).first()

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
            raise marvin.core.exceptions.MarvinError(
                'found a problem while getting DAPall: {0}'.format(str(ee)))

        if response.results['error'] is not None:
            raise MarvinError('found a problem while getting DAPall: {}'
                              .format(str(response.results['error'])))

        data = response.getData()

        return data['dapall_data']
