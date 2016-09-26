# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-09-15 14:50:00
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2016-09-15 16:00:40

from __future__ import print_function, division, absolute_import

import distutils
import warnings

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import marvin
import marvin.core.exceptions
import marvin.tools.spaxel
import marvin.utils.general.general
import marvin.tools.maps

from marvin.api.api import Interaction
from marvin.core import MarvinToolsClass, MarvinError, MarvinUserWarning
from marvin.tools.maps import (__BINTYPES__, __BINTYPES_DEFAULT__,
                               __TEMPLATES_KIN__, __TEMPLATES_KIN_DEFAULT__)


class ModelCube(MarvinToolsClass):
    """A class to interface with MaNGA DAP model cubes.

    This class represents a DAP model cube, initialised either from a file,
    a database, or remotely via the Marvin API.

    Parameters:
        data (``HDUList``, SQLAlchemy object, or None):
            An astropy ``HDUList`` or a SQLAlchemy object of a model cube, to
            be used for initialisation. If ``None``, the normal mode will
            be used (see :ref:`mode-decision-tree`).
        cube (:class:`~marvin.tools.cube.Cube` object)
            The DRP cube object associated with this model cube.
        maps (:class:`~marvin.tools.maps.Maps` object)
            The DAP maps object associated with this model cube. Must match
            the ``bintype``, ``template_kin``, and ``template_pop``.
        filename (str):
            The path of the file containing the model cube to load.
        mangaid (str):
            The mangaid of the model cube to load.
        plateifu (str):
            The plate-ifu of the model cube to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :ref:`mode-decision-tree`.
        bintype (str or None):
            The binning type. For MPL-4, one of the following: ``'NONE',
            'RADIAL', 'STON'`` (if ``None`` defaults to ``'NONE'``).
            For MPL-5 and successive, one of, ``'ALL', 'NRE', 'SPX', 'VOR10'``
            (defaults to ``'ALL'``).
        template_kin (str or None):
            The template use for kinematics. For MPL-4, one of
            ``'M11-STELIB-ZSOL', 'MILES-THIN', 'MIUSCAT-THIN'`` (if ``None``,
            defaults to ``'MIUSCAT-THIN'``). For MPL-5 and successive, the only
            option in ``'GAU-MILESHC'`` (``None`` defaults to it).
        template_pop (str or None):
            A placeholder for a future version in which stellar populations
            are fitted using a different template that ``template_kin``. It
            has no effect for now.
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.
        dapver (str):
            The DAP version to use. Defaults to ``marvin.config.dapver``.

    Return:
        modelcube:
            An object representing the model cube.

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = [
            'data', 'cube', 'maps', 'filename', 'mangaid', 'plateifu', 'mode',
            'drpall', 'drpver', 'dapver', 'bintype', 'template_kin', 'template_pop']

        assert len(args) == 0, 'Maps does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        super(ModelCube, self).__init__(*args, **kwargs)

        # Checks that DAP is at least MPL-5
        MPL5 = distutils.version.StrictVersion('2.0.2')
        if distutils.version.StrictVersion(self._dapver) < MPL5:
            raise MarvinError('ModelCube requires at least dapver=\'2.0.2\'')

        self._cube = kwargs.pop('cube', None)
        self._maps = kwargs.pop('maps', None)

        if kwargs.pop('template_pop', None):
            warnings.warn('template_pop is not yet in use. Ignoring value.', MarvinUserWarning)

        self.bintype = kwargs.pop('bintype', __BINTYPES_DEFAULT__)
        self.template_kin = kwargs.pop('template_kin', __TEMPLATES_KIN_DEFAULT__)
        self.template_pop = None

        assert self.bintype in __BINTYPES__, 'bintype must be on of {0}'.format(__BINTYPES__)
        assert self.template_kin in __TEMPLATES_KIN__, \
            'template_kin must be on of {0}'.format(__TEMPLATES_KIN__)

        self.header = None
        self.wcs = None
        self.shape = None
        self.wavelength = None

        if self.data_origin == 'file':
            self._load_modelcube_from_file()
        elif self.data_origin == 'db':
            self._load_modelcube_from_db()
        elif self.data_origin == 'api':
            self._load_modelcube_from_api()
        else:
            raise MarvinError('data_origin={0} is not valid'.format(self.data_origin))

    def __repr__(self):
        """Representation for ModelCube."""

        return ('<Marvin ModelCube (plateifu={0}, mode={1}, data_origin={2})>'
                .format(repr(self.plateifu), repr(self.mode),
                        repr(self.data_origin)))

    def __getitem__(self, xy):
        """Returns the spaxel for ``(x, y)``"""

        return self.getSpaxel(x=xy[0], y=xy[1], xyorig='lower')

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        daptype = '{0}-{1}'.format(self.bintype, self.template_kin)

        return super(ModelCube, self)._getFullPath('mangadap5', ifu=ifu,
                                                   drpver=self._drpver,
                                                   dapver=self._dapver,
                                                   plate=plate, mode='LOGCUBE',
                                                   daptype=daptype)

    def download(self):
        """Downloads the cube using sdss_access - Rsync"""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        daptype = '{0}-{1}'.format(self.bintype, self.template_kin)

        return super(ModelCube, self).download('mangadap5', ifu=ifu,
                                               drpver=self._drpver,
                                               dapver=self._dapver,
                                               plate=plate, mode='LOGCUBE',
                                               daptype=daptype)

    def _load_modelcube_from_file(self):
        """Initialises a model cube from a file."""

        if self.data is not None:
            assert isinstance(self.data, fits.HDUList), 'data is not an HDUList object'
        else:
            try:
                self.data = fits.open(self.filename)
            except IOError as err:
                raise IOError('filename {0} cannot be found: {1}'.format(self.filename, err))

        self.header = self.data[0].header
        self.shape = self.data['FLUX'].data.shape[1:]
        self.wcs = WCS(self.data['FLUX'].header)
        self.wavelength = self.data['WAVE'].data
        self.redcorr = self.data['REDCORR'].data

        self.plateifu = self.header['PLATEIFU']
        self.mangaid = self.header['MANGAID']

    def _load_modelcube_from_db(self):
        """Initialises a model cube from the DB."""

        mdb = marvin.marvindb
        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise RuntimeError('No db connected')

        else:

            datadb = mdb.datadb
            dapdb = mdb.dapdb

            if self.data:
                assert isinstance(self.data, dapdb.ModelCube), \
                    'data is not an instance of marvindb.dapdb.ModelCube.'
            else:
                # Initial query for version
                version_query = mdb.session.query(dapdb.ModelCube).join(
                    dapdb.File,
                    datadb.PipelineInfo,
                    datadb.PipelineVersion).filter(
                        datadb.PipelineVersion.version == self._dapver).from_self()

                # Query for model cube parameters
                db_modelcube = version_query.join(
                    dapdb.File,
                    datadb.Cube,
                    datadb.IFUDesign).filter(
                        datadb.Cube.plate == plate,
                        datadb.IFUDesign.name == str(ifu)).from_self().join(
                            dapdb.File,
                            dapdb.FileType).filter(dapdb.FileType.value == 'LOGCUBE').join(
                                dapdb.Structure, dapdb.BinType).join(
                                    dapdb.Template,
                                    dapdb.Structure.template_kin_pk == dapdb.Template.pk).filter(
                                        dapdb.BinType.name == self.bintype,
                                        dapdb.Template.name == self.template_kin).all()

                if len(db_modelcube) > 1:
                    raise MarvinError('more than one ModelCube found for '
                                      'this combination of parameters.')

                elif len(db_modelcube) == 0:
                    raise MarvinError('no ModelCube found for this combination of parameters.')

                self.data = db_modelcube[0]

            self.header = self.data.file.hdus[0].header
            self.shape = self.data.file.cube.shape.shape
            self.wcs = WCS(self.data.file.cube.wcs.makeHeader())
            self.wavelength = np.array(self.data.file.cube.wavelength.wavelength, dtype=np.float)
            self.redcorr = np.array(self.data.redcorr[0].value, dtype=np.float)

            self.plateifu = str(self.header['PLATEIFU'].strip())
            self.mangaid = str(self.header['MANGAID'].strip())

    def _load_modelcube_from_api(self):
        """Initialises a model cube from the API."""

        url = marvin.config.urlmap['api']['getModelCube']['url']
        url_full = url.format(name=self.plateifu, bintype=self.bintype,
                              template_kin=self.template_kin)

        try:
            response = Interaction(url_full,
                                   params={'drpver': self._drpver, 'dapver': self._dapver})
        except Exception as ee:
            raise MarvinError('found a problem when checking if remote model cube '
                              'exists: {0}'.format(str(ee)))

        data = response.getData()

        self.header = fits.Header.fromstring(data['header'])
        self.shape = tuple(data['shape'])
        self.wcs = WCS(fits.Header.fromstring(data['wcs_header']))
        self.wavelength = np.array(data['wavelength'])
        self.redcorr = np.array(data['redcorr'])

        self.bintype = data['bintype']
        self.template_kin = data['template_kin']

        self.plateifu = str(self.header['PLATEIFU'].strip())
        self.mangaid = str(self.header['MANGAID'].strip())

    def getSpaxel(self, spectrum=True, properties=True, **kwargs):
        """Returns the |spaxel| matching certain coordinates.

        The coordinates of the spaxel to return can be input as ``x, y`` pixels
        relative to``xyorig`` in the cube, or as ``ra, dec`` celestial
        coordinates.

        If ``spectrum=True``, the returned |spaxel| will be instantiated with the
        DRP spectrum of the spaxel for the DRP cube associated with this
        ModelCube. The same is true for ``properties=True`` for the DAP
        properties of the spaxel in the Maps associated with these coordinates.

        Parameters:
            x,y (int or array):
                The spaxel coordinates relative to ``xyorig``. If ``x`` is an
                array of coordinates, the size of ``x`` must much that of
                ``y``.
            ra,dec (float or array):
                The coordinates of the spaxel to return. The closest spaxel to
                those coordinates will be returned. If ``ra`` is an array of
                coordinates, the size of ``ra`` must much that of ``dec``.
            xyorig ({'center', 'lower'}):
                The reference point from which ``x`` and ``y`` are measured.
                Valid values are ``'center'`` (default), for the centre of the
                spatial dimensions of the cube, or ``'lower'`` for the
                lower-left corner. This keyword is ignored if ``ra`` and
                ``dec`` are defined.
            spectrum (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding DRP spectrum.
            properties (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding DAP properties for this spaxel.

        Returns:
            spaxels (list):
                The |spaxel| objects for this cube/maps corresponding to the
                input coordinates. The length of the list is equal to the
                number of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        kwargs['cube'] = self.cube if spectrum else None
        kwargs['maps'] = self.maps if properties else None
        kwargs['modelcube'] = self

        return marvin.utils.general.general.getSpaxel(**kwargs)

    def _return_extension(self, extension):

        if self.data_origin == 'file':
            return self.data[extension.upper()].data
        elif self.data_origin == 'db':
            return self.data.get3DCube(extension.lower())
        elif self.data_origin == 'api':
            raise MarvinError('cannot return a full cube in remote mode. '
                              'Please use getSpaxel.')

    @property
    def flux(self):
        """Returns the flux extension."""

        return self._return_extension('flux')

    @property
    def ivar(self):
        """Returns the ivar extension."""

        return self._return_extension('ivar')

    @property
    def mask(self):
        """Returns the mask extension."""

        return self._return_extension('mask')

    @property
    def model(self):
        """Returns the model extension."""

        return self._return_extension('model')

    @property
    def emline(self):
        """Returns the emline extension."""

        return self._return_extension('emline')

    @property
    def emline_base(self):
        """Returns the emline_base extension."""

        return self._return_extension('emline_base')

    @property
    def emline_mask(self):
        """Returns the emline_mask extension."""

        return self._return_extension('emline_mask')

    @property
    def stellar_continuum(self):
        """Returns the stellar continuum cube."""

        return (self._return_extension('model') -
                self._return_extension('emline') -
                self._return_extension('emline_base'))

    @property
    def cube(self):
        """Returns the :class:`~marvin.tools.cube.Cube` associated with this ModelCube."""

        if not self._cube:
            if self.data_origin == 'db':
                cube_data = self.data.file.cube
            else:
                cube_data = None

            self._cube = marvin.tools.cube.Cube(data=cube_data,
                                                plateifu=self.plateifu,
                                                drpall=self._drpall,
                                                drpver=self._drpver)

        return self._cube

    @property
    def maps(self):
        """Returns the :class:`~marvin.tools.mas.Maps` associated with this ModelCube."""

        if not self._maps:
            self._maps = marvin.tools.maps.Maps(plateifu=self.plateifu,
                                                bintype=self.bintype,
                                                template_kin=self.template_kin,
                                                drpall=self._drpall,
                                                drpver=self._drpver,
                                                dapver=self._dapver)

        return self._maps
