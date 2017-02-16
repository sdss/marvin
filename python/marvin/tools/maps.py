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
import itertools

import astropy.io.fits
import astropy.wcs
import matplotlib.pyplot as plt
import numpy as np

import marvin
import marvin.api.api
import marvin.core.core
import marvin.core.exceptions
import marvin.tools.cube
import marvin.tools.map
import marvin.tools.spaxel
import marvin.utils.general.general
import marvin.utils.dap
import marvin.utils.dap.bpt
import marvin.utils.six

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None


# The values in the bintypes dictionary for MPL-4 are the execution plan id
# for each bintype.
__BINTYPES_MPL4__ = {'NONE': 3, 'RADIAL': 7, 'STON': 1}
__BINTYPES_MPL4_UNBINNED__ = 'NONE'
__BINTYPES__ = ['ALL', 'NRE', 'SPX', 'VOR10']
__BINTYPES_UNBINNED__ = 'SPX'

__TEMPLATES_KIN_MPL4__ = ['M11-STELIB-ZSOL', 'MIUSCAT-THIN', 'MILES-THIN']
__TEMPLATES_KIN_MPL4_DEFAULT__ = 'MIUSCAT-THIN'
__TEMPLATES_KIN__ = ['GAU-MILESHC']
__TEMPLATES_KIN_DEFAULT__ = 'GAU-MILESHC'

__all__ = ('Maps')


def _is_MPL4(dapver):
    """Returns True if the dapver version is <= MPL-4."""

    assert isinstance(dapver, marvin.utils.six.string_types), 'dapver must be a string'

    if 'v' in dapver:
        dapver = dapver.strip('v').replace('_', '.')

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
        return __BINTYPES_MPL4_UNBINNED__
    else:
        return __BINTYPES_UNBINNED__


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


def _get_bintemps(dapver):
    ''' Get a list of all bin-template types for a given MPL '''

    if _is_MPL4(dapver):
        bins = __BINTYPES_MPL4__.keys()
        temps = __TEMPLATES_KIN_MPL4__
    else:
        bins = __BINTYPES__
        temps = __TEMPLATES_KIN__

    bintemps = ['-'.join(item) for item in list(itertools.product(bins, temps))]
    return bintemps


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
            (defaults to ``'SPX'``).
        template_kin (str or None):
            The template use for kinematics. For MPL-4, one of
            ``'M11-STELIB-ZSOL', 'MILES-THIN', 'MIUSCAT-THIN'`` (if ``None``,
            defaults to ``'MIUSCAT-THIN'``). For MPL-5 and successive, the only
            option in ``'GAU-MILESHC'`` (``None`` defaults to it).
        template_pop (str or None):
            A placeholder for a future version in which stellar populations
            are fitted using a different template that ``template_kin``. It
            has no effect for now.
        nsa_source ({'auto', 'drpall', 'nsa'}):
            Defines how the NSA data for this object should loaded when
            ``Maps.nsa`` is first called. If ``drpall``, the drpall file will
            be used (note that this will only contain a subset of all the NSA
            information); if ``nsa``, the full set of data from the DB will be
            retrieved. If the drpall file or a database are not available, a
            remote API call will be attempted. If ``nsa_source='auto'``, the
            source will depend on how the ``Maps`` object has been
            instantiated. If the cube has ``Maps.data_origin='file'``,
            the drpall file will be used (as it is more likely that the user
            has that file in their system). Otherwise, ``nsa_source='nsa'``
            will be assumed. This behaviour can be modified during runtime by
            modifying the ``Maps.nsa_mode`` with one of the valid values.
        release (str):
            The MPL/DR version of the data to use.

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = [
            'data', 'filename', 'mangaid', 'plateifu', 'mode', 'release',
            'bintype', 'template_kin', 'template_pop', 'nsa_source']

        assert len(args) == 0, 'Maps does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        # For now, we set bintype and template_kin to the kwarg values, so that
        # they can be used by getFullPath.
        self.bintype = kwargs.get('bintype', None)
        self.template_kin = kwargs.get('template_kin', None)

        super(Maps, self).__init__(*args, **kwargs)

        if kwargs.pop('template_pop', None):
            warnings.warn('template_pop is not yet in use. Ignoring value.',
                          marvin.core.exceptions.MarvinUserWarning)

        # We set the bintype  and template_kin again, now using the DAP version
        self.bintype = _get_bintype(self._dapver, bintype=kwargs.pop('bintype', None))
        self.template_kin = _get_template_kin(self._dapver,
                                              template_kin=kwargs.pop('template_kin', None))
        self.template_pop = None

        self.header = None
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

        self._check_versions(self)

    def __repr__(self):
        return ('<Marvin Maps (plateifu={0.plateifu!r}, mode={0.mode!r}, '
                'data_origin={0.data_origin!r}, bintype={0.bintype}, '
                'template_kin={0.template_kin})>'.format(self))

    def __getitem__(self, value):
        """Gets either a spaxel or a map depending on the type on input."""

        if isinstance(value, tuple):
            assert len(value) == 2, 'slice must have two elements.'
            x, y = value
            return self.getSpaxel(x=x, y=y, xyorig='lower')
        elif isinstance(value, marvin.utils.six.string_types):
            parsed_property = self.properties.get(value)
            if parsed_property is None:
                raise marvin.core.exceptions.MarvinError('invalid property')
            maps_property, channel = parsed_property
            return self.getMap(maps_property.name, channel=channel)
        else:
            raise marvin.core.exceptions.MarvinError('invalid type for getitem.')

    @staticmethod
    def _check_versions(instance):
        """Confirm that drpver and dapver match the ones from the header.

        This is written as a staticmethod because we'll also use if for
        ModelCube.

        """

        header_drpver = instance.header['VERSDRP3']
        isMPL4 = False
        if instance._release == 'MPL-4' and header_drpver == 'v1_5_0':
            header_drpver = 'v1_5_1'
            isMPL4 = True
        assert header_drpver == instance._drpver, ('mismatch between maps._drpver={0} '
                                                   'and header drpver={1}'
                                                   .format(instance._drpver, header_drpver))

        # MPL-4 does not have VERSDAP
        if isMPL4:
            assert 'VERSDAP' not in instance.header, 'mismatch between maps._dapver and header'
        else:
            header_dapver = instance.header['VERSDAP']
            assert header_dapver == instance._dapver, 'mismatch between maps._dapver and header'

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        params = self._getPathParams()
        path_type = params.pop('path_type')

        return super(Maps, self)._getFullPath(path_type, **params)

    def download(self):
        """Downloads the maps using sdss_access - Rsync"""

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

        bintype = _get_bintype(self._dapver, bintype=self.bintype)
        template_kin = _get_template_kin(self._dapver, template_kin=self.template_kin)

        if _is_MPL4(self._dapver):
            niter = int('{0}{1}'.format(__TEMPLATES_KIN_MPL4__.index(template_kin),
                                        __BINTYPES_MPL4__[bintype]))
            params = dict(drpver=self._drpver, dapver=self._dapver,
                          plate=plate, ifu=ifu, bintype=bintype, n=niter,
                          path_type='mangamap')
        else:
            daptype = '{0}-{1}'.format(bintype, template_kin)
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

        self.header = self.data[0].header

        self.mangaid = self.header['MANGAID'].strip()
        self.plateifu = self.header['PLATEIFU'].strip()

        # We use EMLINE_GFLUX because is present in MPL-4 and 5 and is not expected to go away.
        header = self.data['EMLINE_GFLUX'].header
        naxis = header['NAXIS']
        wcs_pre = astropy.wcs.WCS(header)
        # Takes only the first two axis.
        self.wcs = wcs_pre.sub(2) if naxis > 2 else naxis
        self.shape = self.data['EMLINE_GFLUX'].data.shape[-2:]

        # Checks and populates release.
        file_drpver = self.header['VERSDRP3']
        file_drpver = 'v1_5_1' if file_drpver == 'v1_5_0' else file_drpver

        file_ver = marvin.config.lookUpRelease(file_drpver)
        assert file_ver is not None, 'cannot find file version.'

        if file_ver != self._release:
            warnings.warn('mismatch between file version={0} and object release={1}. '
                          'Setting object release to {0}'.format(file_ver, self._release),
                          marvin.core.exceptions.MarvinUserWarning)
            self._release = file_ver

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)

        # Checks the bintype and template_kin from the header
        if not _is_MPL4(self._dapver):
            header_bintype = self.data[0].header['BINKEY'].strip().upper()
            header_bintype = 'SPX' if header_bintype == 'NONE' else header_bintype
        else:
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
            raise marvin.core.exceptions.MarvinError('No db connected')

        if sqlalchemy is None:
            raise marvin.core.exceptions.MarvinError('sqlalchemy required to access the local DB.')

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

        self.header = self.data.primary_header

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
            response = self.ToolInteraction(url_full)
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when checking if remote maps exists: {0}'.format(str(ee)))

        data = response.getData()

        if self.plateifu not in data:
            raise marvin.core.exceptions.MarvinError('remote maps has a different plateifu!')

        self.header = astropy.io.fits.Header.fromstring(data[self.plateifu]['header'])

        # Sets the mangaid
        self.mangaid = data[self.plateifu]['mangaid']

        # Gets the shape from the associated cube.
        self.shape = data[self.plateifu]['shape']

        # Sets the WCS
        self.wcs = astropy.wcs.WCS(astropy.io.fits.Header.fromstring(data[self.plateifu]['wcs']))

        return

    @property
    def cube(self):
        """Returns the :class:`~marvin.tools.cube.Cube` for with this Maps."""

        if not self._cube:
            try:
                cube = marvin.tools.cube.Cube(plateifu=self.plateifu,
                                              release=self._release)
            except Exception as err:
                raise marvin.core.exceptions.MarvinError(
                    'cannot instantiate a cube for this Maps. Error: {0}'.format(err))
            self._cube = cube

        return self._cube

    def getSpaxel(self, x=None, y=None, ra=None, dec=None,
                  spectrum=True, modelcube=False, **kwargs):
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
            modelcube (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding ModelCube data.


        Returns:
            spaxels (list):
                The |spaxel| objects for this cube/maps corresponding to the
                input coordinates. The length of the list is equal to the
                number of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        kwargs['cube'] = self.cube if spectrum else False
        kwargs['maps'] = self.get_unbinned()
        kwargs['modelcube'] = modelcube

        return marvin.utils.general.general.getSpaxel(x=x, y=y, ra=ra, dec=dec, **kwargs)

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

        # TODO: do the error propogation (BHA)
        map_1.ivar = None

        map_1.mask &= map_2.mask

        map_1.channel = '{0}/{1}'.format(channel_1, channel_2)

        if map_1.unit != map_2.unit:
            map_1.unit = '{0}/{1}'.format(map_1.unit, map_2.unit)
        else:
            map_1.unit = ''

        return map_1

    def is_binned(self):
        """Returns True if the Maps is not unbinned."""

        if _is_MPL4(self._dapver):
            return self.bintype != __BINTYPES_MPL4_UNBINNED__
        else:
            return self.bintype != __BINTYPES_UNBINNED__

    def get_unbinned(self):
        """Returns a version of ``self`` corresponding to the unbinned Maps."""

        if _is_MPL4(self._dapver):
            unbinned_name = __BINTYPES_MPL4_UNBINNED__
        else:
            unbinned_name = __BINTYPES_UNBINNED__

        if self.bintype == unbinned_name:
            return self
        else:
            return Maps(plateifu=self.plateifu, release=self._release, bintype=unbinned_name,
                        template_kin=self.template_kin, template_pop=self.template_pop,
                        mode=self.mode)

    def get_bin_spaxels(self, binid, load=False, only_list=False):
        """Returns the list of spaxels belonging to a given ``binid``.

        If ``load=True``, the spaxel objects are loaded. Otherwise, they can be
        initiated by doing ``Spaxel.load()``. If ``only_list=True``, the method
        will return just a tuple containing the x and y coordinates of the spaxels.

        """

        if self.data_origin == 'file':
            spaxel_coords = zip(*np.where(self.data['BINID'].data.T == binid))

        elif self.data_origin == 'db':
            mdb = marvin.marvindb

            if _is_MPL4(self._dapver):
                table = mdb.dapdb.SpaxelProp
            else:
                table = mdb.dapdb.SpaxelProp5

            spaxel_coords = mdb.session.query(table.x, table.y).join(mdb.dapdb.File).filter(
                table.binid == binid, mdb.dapdb.File.pk == self.data.pk).order_by(
                    table.x, table.y).all()

        elif self.data_origin == 'api':
            url = marvin.config.urlmap['api']['getbinspaxels']['url']

            url_full = url.format(name=self.plateifu,
                                  bintype=self.bintype,
                                  template_kin=self.template_kin,
                                  binid=binid)

            try:
                response = self.ToolInteraction(url_full)
            except Exception as ee:
                raise marvin.core.exceptions.MarvinError(
                    'found a problem requesting the spaxels for binid={0}: {1}'
                    .format(binid, str(ee)))

            response = response.getData()
            spaxel_coords = response['spaxels']

        spaxel_coords = list(spaxel_coords)
        if len(spaxel_coords) == 0:
            return []
        else:
            if only_list:
                return tuple([tuple(cc) for cc in spaxel_coords])

        spaxels = [marvin.tools.spaxel.Spaxel(x=cc[0], y=cc[1], maps=self, load=load)
                   for cc in spaxel_coords]

        return spaxels

    def get_bpt(self, method='kewley06', snr=3, return_figure=True, show_plot=True, use_oi=True):
        """Returns the BPT diagram for this target.

        This method calculates the BPT diagram for this target using emission line maps and
        returns a dictionary of classification masks, that can be used to select spaxels
        that have been classified as belonging to a certain excitation process. It also provides
        plotting functionalities.

        Parameters:
            method ({'kewley06'}):
                The method used to determine the boundaries between different excitation
                mechanisms. Currently, the only available method is ``'kewley06'``, based on
                Kewley et al. (2006). Other methods may be added in the future. For a detailed
                explanation of the implementation of the method check the
                :ref:`BPT documentation <marvin-bpt>`.
            snr (float or dict):
                The signal-to-noise cutoff value for the emission lines used to generate the BPT
                diagram. If ``snr`` is a single value, that signal-to-noise will be used for all
                the lines. Alternatively, a dictionary of signal-to-noise values, with the
                emission line channels as keys, can be used.
                E.g., ``snr={'ha': 5, 'nii': 3, 'oi': 1}``. If some values are not provided,
                they will default to ``SNR>=3``.
            return_figure (bool):
                If ``True``, it also returns the matplotlib figure_ of the BPT diagram plot,
                which can be used to modify the style of the plot.
            show_plot (bool):
                If ``True``, interactively display the BPT plot.
            use_oi (bool):
                If ``True``, turns uses the OI diagnostic line in classifying BPT spaxels

        Returns:
            bpt_return:
                ``get_bpt`` always returns a dictionary of classification masks. These
                classification masks (not to be confused with bitmasks) are boolean arrays with the
                same shape as the Maps or Cube (without the spectral dimension) that can be used
                to select spaxels belonging to a certain excitation process (e.g., star forming).
                The keys of the dictionary, i.e., the classification categories, may change
                depending on the selected `method`. Consult the :ref:`BPT <marvin-bpt>`
                documentation for more details.
                If ``return_figure=True``, ``get_bpt`` will return a tuple, the first elemnt of
                which is the dictionary of classification masks, and the second the matplotlib
                figure.

        Example:
            >>> cube = Cube(plateifu='8485-1901')
            >>> maps = cube.getMaps()
            >>> bpt_masks, bpt_figure = maps.get_bpt(snr=5, return_figure=True, show_plot=False)

            Now we can use the masks to select star forming spaxels from the cube

            >>> sf_spaxels = cube.flux[bpt_masks['sf']]

            And we can save the figure as a PDF

            >>> bpt_figure.savefig('8485_1901_bpt.pdf')

        .. _figure: http://matplotlib.org/api/figure_api.html

        """

        # Makes sure all the keys in the snr keyword are lowercase
        if isinstance(snr, dict):
            snr = dict((kk.lower(), vv) for kk, vv in snr.items())

        # If we don't want the figure but want to show the plot, we still need to
        # temporarily get it.
        do_return_figure = True if return_figure or show_plot else False

        # Disables ion() if we are not showing the plot.
        plt_was_interactive = plt.isinteractive()
        if not show_plot and plt_was_interactive:
            plt.ioff()

        bpt_return = marvin.utils.dap.bpt.bpt_kewley06(self, snr=snr,
                                                       return_figure=do_return_figure,
                                                       use_oi=use_oi)

        if show_plot:
            plt.ioff()
            plt.show()

        # Restores original ion() status
        if plt_was_interactive and not plt.isinteractive():
            plt.ion()

        # Returs what we actually asked for.
        if return_figure and do_return_figure:
            return bpt_return
        elif not return_figure and do_return_figure:
            return bpt_return[0]
        else:
            return bpt_return
