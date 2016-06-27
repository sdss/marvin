#!/usr/bin/env python
# encoding: utf-8
#
# maps.py
#
# Created by José Sánchez-Gallego on 20 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import warnings

import astropy.io.fits
import astropy.wcs

import marvin
import marvin.api.api
import marvin.core.core
import marvin.core.exceptions
import marvin.tools.cube
import marvin.tools.map
import marvin.utils.general.general

try:
    import sqlalchemy
except:
    sqlalchemy = None


class Maps(marvin.core.core.MarvinToolsClass):
    """Returns an object representing a DAP Maps file.

    Parameters:
        filename (str):
            The path of the file containing the data cube to load.
        mangaid (str):
            The mangaid of the data cube to load.
        plateifu (str):
            The plate-ifu of the data cube to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :doc:`mode-decision-tree>`.
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.
        dapver (str):
            The DAP version to use. Defaults to ``marvin.config.dapver``.
        bintype (str or None):
            The binning type of the DAP MAPS file to use. The default value is
            ``'NONE'``
        niter (int or None):
            The iteration number of the DAP map.
        load_drp (bool):
            If ``True`` (default), loads the DRP cube for this map. Spaxels
            extracted from the map will be loaded with the DRP spectrum.

    """

    def __init__(self, *args, **kwargs):

        # TODO: We want to be able to open a Maps from the names of its
        # template, bintype, etc, instead of using niter, which is not very
        # informative.

        # TODO: this is not the nicest implementation possible, but we do want
        # to avoid the user using wrong keywords without getting alerted.
        # Python 3 allows to do "def __init__(self, *args, kw1=None, kw2=None, **kwargs)"
        list_of_valid_kwargs = [
            'filename', 'mangaid', 'plateifu', 'mode', 'drpall', 'drpver',
            'dapver', 'bintype', 'niter', 'load_drp']

        assert len(args) == 0, 'Maps does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in list_of_valid_kwargs, 'keyword {0} is not valid'.format(kw)

        self.bintype = kwargs.pop('bintype', None)
        self.niter = kwargs.pop('niter', None)
        self._load_drp = kwargs.pop('load_drp', None)

        # Either bintype and niter are None, or they have values
        assert ((self.bintype is None and self.niter is None) or
                (self.bintype is not None and self.niter is not None)), \
            'bintype and niter must be both None or both not None'

        self.wcs = None
        self.data = None
        self.cube = None
        self.wcs = None
        self.shape = None
        self.template_kin = None

        super(Maps, self).__init__(*args, **kwargs)

        if self.data_origin == 'file':
            self._load_maps_from_file()
        elif self.data_origin == 'db':
            self._load_maps_from_db()
        elif self.data_origin == 'api':
            self._load_maps_from_api()
        else:
            raise marvin.core.exceptions.MarvinError(
                'data_origin={0} is not valid'.format(self.data_origin))

        # Loads the cube
        if self._load_drp:
            try:
                self._load_drp_cube(**kwargs)
            except Exception as ee:
                warnings.warn('failed loading DRP cube for Maps with plateifu={0}. '
                              'Error message is: {1}'.format(self.plateifu, str(ee)),
                              marvin.core.exceptions.MarvinUserWarning)

    def __repr__(self):
        return ('<Marvin Maps (plateifu={0.plateifu}, mode={0.mode}, '
                'data_origin={0.data_origin})>'.format(self))

    def _getPathParams(self):
        """Returns a dictionary with the paramters of the Maps file.

        The parameters that define a Maps file depend on whether it is a
        ``mangamap`` or a ``mangadefault``. If ``bintype`` and ``niter``
        have been defined in the initilisatation of the object, we return
        the parameters of a ``mangamap``, otherwise the ones for a
        ``mangadefault``.

        The output of this class is mostly intended to be used by
        :func:`Maps._getFullPath` and :func:`Maps.download`.

        """

        plate, ifu = self.plateifu.split('-')

        if self.bintype and self.niter:
            path_type = 'mangamap'
            params = dict(drpver=self._drpver, dapver=self._dapver,
                          plate=plate, ifu=ifu, bintype=self.bintype,
                          n=self.niter, mode='CUBE')
        else:
            path_type = 'mangadefault'
            params = dict(drpver=self._drpver, dapver=self._dapver,
                          plate=plate, ifu=ifu, mode='CUBE')

        params['path_type'] = path_type

        return params

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        params = self._getPathParams()
        path_type = params.pop('path_type')

        return super(Maps, self)._getFullPath(path_type, **params)

    def download(self, **kwargs):
        """Downloads the cube using sdss_access - Rsync"""

        if not self.plateifu:
            return None

        params = self._getPathParams()
        path_type = params.pop('path_type')

        return super(Maps, self)._getFullPath(path_type, **params)

    def _load_maps_from_file(self):
        """Loads a MAPS file."""

        self.data = astropy.io.fits.open(self.filename)
        self.mangaid = self.data[0].header['MANGAID'].strip()
        self.plateifu = self.data[0].header['PLATEIFU'].strip()

        # Loads WCS and shape from the first N-dimensional extension.
        # TODO: this may break if the datamodel changes. There may be a better
        # way of implementing this.
        for ext in self.data:
            header = ext.header
            if 'NAXIS' in header and header['NAXIS'] >= 2:
                naxis = header['NAXIS']
                wcs_pre = astropy.wcs.WCS(header)
                # Takes only the first two axis.
                self.wcs = wcs_pre.sub(2) if naxis > 2 else naxis
                self.shape = ext.data.shape[-2:]

        # Sets the bintype from the header
        self.bintype = self.data[0].header['BINTYPE']

        # Sets the template
        self.template_kin = self.data[0].header['TPLKEY'].upper()

    def _load_maps_from_db(self):
        """Loads the ``mangadap.File`` object for this Maps."""

        mdb = marvin.marvindb

        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise RuntimeError('No db connected')

        if sqlalchemy is None:
            raise RuntimeError('sqlalchemy required to access the local DB.')

        version_query = mdb.session.query(mdb.dapdb.File).join(
            mdb.datadb.PipelineInfo,
            mdb.datadb.PipelineVersion).filter(
                mdb.datadb.PipelineVersion.version == self._dapver).from_self()

        # Runs the query if we are selecting a default map.
        if self.bintype is None and self.niter is None:
            db_maps_file = version_query.join(
                mdb.datadb.Cube,
                mdb.datadb.IFUDesign,
                mdb.dapdb.CurrentDefault).filter(
                    mdb.datadb.Cube.plate == plate,
                    mdb.datadb.IFUDesign.name == str(ifu)).all()

        # Otherwise
        else:
            # Splits niter into its parts.
            niter_str = str('{0:>03}'.format(self.niter))
            __, template_kin_id, execution_plan_id = [ii for ii in niter_str]

            db_maps_file = version_query.join(
                mdb.datadb.Cube,
                mdb.datadb.IFUDesign).filter(
                    mdb.datadb.Cube.plate == plate,
                    mdb.datadb.IFUDesign.name == str(ifu)).from_self().join(
                        mdb.dapdb.Structure,
                        mdb.dapdb.BinType,
                        mdb.dapdb.ExecutionPlan).join(
                            mdb.dapdb.Template,
                            mdb.dapdb.Structure.template_kin_pk == mdb.dapdb.Template.pk).filter(
                                sqlalchemy.func.upper(mdb.dapdb.BinType.name) ==
                                sqlalchemy.func.upper(self.bintype),
                                mdb.dapdb.ExecutionPlan.id == execution_plan_id,
                                mdb.dapdb.Template.id == template_kin_id).all()

        if len(db_maps_file) > 1:
            raise marvin.core.exceptions.MarvinError(
                'more than one Maps file found for this combination of parameters.')
        elif len(db_maps_file) == 0:
            raise marvin.core.exceptions.MarvinError(
                'no Maps file found for this combination of parameters.')

        self.data = db_maps_file[0]

        # Gets the cube header
        cubehdr = self.data.cube.header

        # Gets the mangaid
        self.mangaid = cubehdr['MANGAID'].strip()

        # Gets the shape from the associated cube.
        self.shape = self.data.cube.shape.shape

        # Defines the bintype and template.
        self.bintype = self.data.structure.bintype.name.upper()
        self.template_kin = self.data.structure.template_kin.name.upper()

        # Creates the WCS from the cube's WCS header
        # TODO: I'm assuming the cube WCS is the same as the maps. That might
        # not be true (JSG)
        self.wcs = astropy.wcs.WCS(self.data.cube.wcs.makeHeader())

    def _load_maps_from_api(self):
        """Loads a Maps object from remote."""

        url = marvin.config.urlmap['api']['getMaps']['url']

        try:
            response = marvin.api.api.Interaction(url.format(name=self.plateifu))
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when checking if remote cube exists: {0}'
                .format(str(ee)))

        data = response.getData()

        if self.plateifu not in data:
            raise marvin.core.exceptions.MarvinError(
                'remote cube has a different plateifu!')

        # TODO: replace self.data with a property that returns an error for
        # Maps initialised from remote.
        self.data = None

        # Sets the mangaid
        self.mangaid = data[self.plateifu]['mangaid']

        # Gets the shape from the associated cube.
        self.shape = data[self.plateifu]['shape']

        # Defines the bintype and template.
        self.bintype = data[self.plateifu]['bintype']
        self.template_kin = data[self.plateifu]['template_kin']

        # Sets the WCS
        self.wcs = data[self.plateifu]['wcs']

        return

    def _load_drp_cube(self, **kwargs):
        """Loads the DRP cube associated to this maps file."""

        drpver = kwargs.get('drpver', None)
        drpall = kwargs.get('drpall', None)

        self.cube = marvin.tools.cube.Cube(plateifu=self.plateifu,
                                           drpver=drpver, drpall=drpall,
                                           mode=self.mode)

    def getSpaxel(self, **kwargs):
        """Returns the |spaxel| matching certain coordinates.

        The coordinates of the spaxel to return can be input as ``x, y`` pixels
        relative to``xyorig`` in the cube, or as ``ra, dec`` celestial
        coordinates.

        If the ``Maps`` object has been initialised with ``load_drp=True``,
        the spaxels returned will contain the DRP spaxel, otherwise they will
        be loaded only with DAP parameters.

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

        Returns:
            spaxels (list):
                The |spaxel| objects for this cube/maps corresponding to the
                input coordinates. The length of the list is equal to the
                number of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        kwargs['cube_object'] = self.cube
        kwargs['maps_object'] = self

        return marvin.utils.general.general.getSpaxel(**kwargs)

    def getMap(self, category, channel=None):
        """Retrieves a :class:~marvin.tools.map.Map object.

        Parameters:
            category (str):
                The category of the map to be extractred. E.g., `'EMLINE_GFLUX'`.
            channel (str or None):
                If the ``category`` contains multiple channels, the channel to use,
                e.g., ``Ha-6564'. Otherwise, ``None``.

        """

        return marvin.tools.map.Map(self, category, channel=channel)
