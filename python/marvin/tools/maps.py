#!/usr/bin/env python
# encoding: utf-8
#
# maps.py
#
# Created by José Sánchez-Gallego on 20 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import distutils.version
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
import marvin.utils.dap
import marvin.utils.six

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None


# The values in the bintypes dictionary for MPL-4 are the execution plan id
# for each bintype.
__BINTYPES_MPL4__ = {'NONE': 3, 'RADIAL': 7, 'STON': 1}
__BINTYPES_MPL4_DEFAULT__ = 'NONE'
__BINTYPES__ = ['ALL', 'NRE', 'SPX', 'VOR10']
__BINTYPES_DEFAULT__ = 'SPX'

__TEMPLATES_KIN_MPL4__ = ['M11-STELIB-ZSOL', 'MIUSCAT-THIN', 'MILES-THIN']
__TEMPLATES_KIN_MPL4_DEFAULT__ = 'MIUSCAT-THIN'
__TEMPLATES_KIN__ = ['GAU-MILESHC']
__TEMPLATES_KIN_DEFAULT__ = 'GAU-MILESHC'

__all__ = ('Maps')


def _is_MPL4(dapver):
    """Returns True if the dapver version is <= MPL-4."""

    dap_version = distutils.version.StrictVersion(dapver)
    MPL4_version = distutils.version.StrictVersion('1.1.1')

    return dap_version <= MPL4_version


def _get_bintype(dapver, bintype=None):
    """Checks the bintype and returns the default value if None."""

    if bintype is not None:
        bintype = bintype.upper()
        bintypes_check = __BINTYPES_MPL4__.keys() if _is_MPL4(dapver) else __BINTYPES__
        assert bintype in bintypes_check, ('invalid bintype. bintype must be one of {0}'
                                           .format(bintypes_check))
        return bintype

    # Defines the default value depending on the version
    if _is_MPL4(dapver):
        return __BINTYPES_MPL4_DEFAULT__
    else:
        return __BINTYPES_DEFAULT__


def _get_template_kin(dapver, template_kin=None):
    """Checks the template_kin and returns the default value if None."""

    if template_kin is not None:
        template_kin = template_kin.upper()
        templates_check = __TEMPLATES_KIN_MPL4__ if _is_MPL4(dapver) else __TEMPLATES_KIN__
        assert template_kin in templates_check, ('invalid template_kin. '
                                                 'template_kin must be one of {0}'
                                                 .format(templates_check))
        return template_kin

    # Defines the default value depending on the version
    if _is_MPL4(dapver):
        return __TEMPLATES_KIN_MPL4_DEFAULT__
    else:
        return __TEMPLATES_KIN_DEFAULT__


class Maps(marvin.core.core.MarvinToolsClass):
    """Returns an object representing a DAP Maps file.

    Parameters:
        data (``HDUList``, SQLAlchemy object, or None):
            An astropy ``HDUList`` or a SQLAlchemy object of a maps, to
            be used for initialisation. If ``None``, the normal mode will
            be used (see :ref:`mode-decision-tree`).
        filename (str):
            The path of the data cube file containing the spaxel to load.
        mangaid (str):
            The mangaid of the spaxel to load.
        plateifu (str):
            The plate-ifu of the spaxel to load (either ``mangaid`` or
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

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = [
            'data', 'filename', 'mangaid', 'plateifu', 'mode', 'drpall',
            'drpver', 'dapver', 'bintype', 'template_kin', 'template_pop']

        assert len(args) == 0, 'Maps does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        super(Maps, self).__init__(*args, **kwargs)

        if kwargs.pop('template_pop', None):
            warnings.warn('template_pop is not yet in use. Ignoring value.',
                          marvin.core.exceptions.MarvinUserWarning)

        self.bintype = _get_bintype(self._dapver, bintype=kwargs.pop('bintype', None))
        self.template_kin = _get_template_kin(self._dapver,
                                              template_kin=kwargs.pop('template_kin', None))
        self.template_pop = None

        self.wcs = None
        self.shape = None
        self._cube = None

        if self.data_origin == 'file':
            self._load_maps_from_file(data=self.data)
        elif self.data_origin == 'db':
            self._load_maps_from_db(data=self.data)
        elif self.data_origin == 'api':
            self._load_maps_from_api()
        else:
            raise marvin.core.exceptions.MarvinError(
                'data_origin={0} is not valid'.format(self.data_origin))

        self.properties = marvin.utils.dap.get_dap_datamodel(self._dapver)

    def __repr__(self):
        return ('<Marvin Maps (plateifu={0.plateifu!r}, mode={0.mode!r}, '
                'data_origin={0.data_origin!r})>'.format(self))

    def __getitem__(self, value):
        """Gets either a spaxel or a map depending on the type on input."""

        if isinstance(value, tuple):
            assert len(value) == 2, 'slice must have two elements.'
            x, y = value
            return self.getSpaxel(x=x, y=y, xyorig='lower')
        elif isinstance(value, marvin.utils.six.string_types):
            parsed_property = self.properties.get(value)
            if parsed_property is None:
                raise marvin.core.MarvinError('invalid property')
            maps_property, channel = parsed_property
            return self.getMap(maps_property.name, channel=channel)
        else:
            raise marvin.core.MarvinError('invalid type for getitem.')

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        params = self._getPathParams()
        path_type = params.pop('path_type')

        return super(Maps, self)._getFullPath(path_type, **params)

    def download(self):
        """Downloads the cube using sdss_access - Rsync"""

        if not self.plateifu:
            return None

        params = self._getPathParams()
        path_type = params.pop('path_type')

        return super(Maps, self).download(path_type, **params)

    def _getPathParams(self):
        """Returns a dictionary with the paramters of the Maps file.

        The output of this class is mostly intended to be used by
        :func:`Maps._getFullPath` and :func:`Maps.download`.

        """

        plate, ifu = self.plateifu.split('-')

        if _is_MPL4(self._dapver):
            niter = int('{0}{1}'.format(__TEMPLATES_KIN_MPL4__.index(self.template_kin),
                                        __BINTYPES_MPL4__[self.bintype]))
            params = dict(drpver=self._drpver, dapver=self._dapver,
                          plate=plate, ifu=ifu, bintype=self.bintype, n=niter,
                          path_type='mangadap')
        else:
            daptype = '{0}-{1}'.format(self.bintype, self.template_kin)
            params = dict(drpver=self._drpver, dapver=self._dapver,
                          plate=plate, ifu=ifu, mode='MAPS', daptype=daptype,
                          path_type='mangadap5')

        return params

    def _load_maps_from_file(self, data=None):
        """Loads a MAPS file."""

        if data is not None:
            assert isinstance(data, astropy.io.fits.HDUList), 'data is not a HDUList.'
        else:
            self.data = astropy.io.fits.open(self.filename)

        self.mangaid = self.data[0].header['MANGAID'].strip()
        self.plateifu = self.data[0].header['PLATEIFU'].strip()

        # Check DRP and DAP versions of the file.
        try:
            header_drpver = self.data[0].header['VERSDRP3']
            # There is an inconsistency between the drpver officially considered MPL-4
            # and the version in the header. We fix it here.
            if header_drpver == 'v1_5_0':
                header_drpver = 'v1_5_1'
            if header_drpver != self._drpver:
                self._drpver = header_drpver
        except KeyError:
            raise marvin.core.exceptions.MarvinError('cannot retrieve DRP version from file.')

        if self._drpver == 'v1_5_1':
            if self._dapver != '1.1.1':
                self._dapver = '1.1.1'
        else:
            try:
                header_dapver = self.data[0].header['VERSDAP']
                if header_dapver != self._dapver:
                    self._dpver = header_dapver
            except KeyError:
                raise marvin.core.exceptions.MarvinError('cannot retrieve DAP version from file.')

        # We use EMLINE_GFLUX because is present in MPL-4 and 5 and is not expected to go away.
        header = self.data['EMLINE_GFLUX'].header
        naxis = header['NAXIS']
        wcs_pre = astropy.wcs.WCS(header)
        # Takes only the first two axis.
        self.wcs = wcs_pre.sub(2) if naxis > 2 else naxis
        self.shape = self.data['EMLINE_GFLUX'].data.shape[-2:]

        # Checks the bintype and template_kin from the header
        header_bintype = self.data[0].header['BINTYPE'].strip().upper()
        header_template_kin_key = 'TPLKEY' if _is_MPL4(self._dapver) else 'SCKEY'
        header_template_kin = self.data[0].header[header_template_kin_key].strip().upper()

        if self.bintype != header_bintype:
            self.bintype = header_bintype

        if self.template_kin != header_template_kin:
            self.template_kin = header_template_kin

    def _load_maps_from_db(self, data=None):
        """Loads the ``mangadap.File`` object for this Maps."""

        mdb = marvin.marvindb

        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise RuntimeError('No db connected')

        if sqlalchemy is None:
            raise RuntimeError('sqlalchemy required to access the local DB.')

        if data is not None:
            assert isinstance(data, mdb.dapdb.File), 'data in not a marvindb.dapdb.File object.'
        else:

            datadb = mdb.datadb
            dapdb = mdb.dapdb
            # Initial query for version
            version_query = mdb.session.query(dapdb.File).join(
                datadb.PipelineInfo,
                datadb.PipelineVersion).filter(
                    datadb.PipelineVersion.version == self._dapver).from_self()

            # Query for maps parameters
            db_maps_file = version_query.join(
                datadb.Cube,
                datadb.IFUDesign).filter(
                    datadb.Cube.plate == plate,
                    datadb.IFUDesign.name == str(ifu)).from_self().join(
                        dapdb.FileType).filter(dapdb.FileType.value == 'MAPS').join(
                            dapdb.Structure, dapdb.BinType).join(
                                dapdb.Template,
                                dapdb.Structure.template_kin_pk == dapdb.Template.pk).filter(
                                    dapdb.BinType.name == self.bintype,
                                    dapdb.Template.name == self.template_kin).all()

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

        # Creates the WCS from the cube's WCS header
        self.wcs = astropy.wcs.WCS(self.data.cube.wcs.makeHeader())

    def _load_maps_from_api(self):
        """Loads a Maps object from remote."""

        url = marvin.config.urlmap['api']['getMaps']['url']

        url_full = url.format(name=self.plateifu,
                              bintype=self.bintype,
                              template_kin=self.template_kin)

        try:
            response = marvin.api.api.Interaction(url_full, params={'drpver': self._drpver,
                                                                    'dapver': self._dapver})
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when checking if remote maps exists: {0}'.format(str(ee)))

        data = response.getData()

        if self.plateifu not in data:
            raise marvin.core.exceptions.MarvinError('remote maps has a different plateifu!')

        # Sets the mangaid
        self.mangaid = data[self.plateifu]['mangaid']

        # Gets the shape from the associated cube.
        self.shape = data[self.plateifu]['shape']

        # Sets the WCS
        self.wcs = data[self.plateifu]['wcs']

        return

    @property
    def cube(self):
        """Returns the :class:`~marvin.tools.cube.Cube` for with this Maps."""

        if not self._cube:
            try:
                cube = marvin.tools.cube.Cube(plateifu=self.plateifu, drpver=self._drpver)
            except Exception as err:
                raise marvin.core.exceptions.MarvinError(
                    'cannot instantiate a cube for this Maps. Error: {0}'.format(err))
            self._cube = cube

        return self._cube

    def getSpaxel(self, spectrum=True, **kwargs):
        """Returns the |spaxel| matching certain coordinates.

        The coordinates of the spaxel to return can be input as ``x, y`` pixels
        relative to``xyorig`` in the cube, or as ``ra, dec`` celestial
        coordinates.

        If ``spectrum=True``, the returned |spaxel| will be instantiated with the
        DRP spectrum of the spaxel for the DRP cube associated with this Maps.

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

        Returns:
            spaxels (list):
                The |spaxel| objects for this cube/maps corresponding to the
                input coordinates. The length of the list is equal to the
                number of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        kwargs['cube'] = self.cube if spectrum else None
        kwargs['maps'] = self

        return marvin.utils.general.general.getSpaxel(**kwargs)

    def getMap(self, property_name, channel=None):
        """Retrieves a :class:`~marvin.tools.map.Map` object.

        Parameters:
            property_name (str):
                The property of the map to be extractred.
                E.g., `'emline_gflux'`.
            channel (str or None):
                If the ``property`` contains multiple channels,
                the channel to use, e.g., ``ha_6564'. Otherwise, ``None``.

        """

        return marvin.tools.map.Map(self, property_name, channel=channel)

    def getMapRatio(self, property_name, channel_1, channel_2):
        """Returns a ratio :class:`~marvin.tools.map.Map`.

        For a given ``property_name``, returns a :class:`~marvin.tools.map.Map`
        which is the ratio of ``channel_1/channel_2``.

        Parameters:
            property_name (str):
                The property_name of the map to be extractred.
                E.g., `'emline_gflux'`.
            channel_1,channel_2 (str):
                The channels to use.

        """

        map_1 = self.getMap(property_name, channel=channel_1)
        map_2 = self.getMap(property_name, channel=channel_2)

        map_1.value /= map_2.value

        # TODO: this is probably wrong (JSG)
        map_1.ivar /= map_2.ivar

        map_1.mask &= map_2.mask

        map_1.channel = '{0}/{1}'.format(channel_1, channel_2)

        return map_1
