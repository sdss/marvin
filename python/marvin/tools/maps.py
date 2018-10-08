#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-11-08
# @Filename: maps.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-08-23 11:53:04


from __future__ import absolute_import, division, print_function

import copy
import inspect
import warnings

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
import six

import marvin
import marvin.api.api
import marvin.core.exceptions
import marvin.tools.cube
import marvin.tools.modelcube
import marvin.tools.quantities.map
import marvin.tools.spaxel
import marvin.utils.dap.bpt
import marvin.utils.general.general
from marvin.utils.datamodel.dap import datamodel
from marvin.utils.datamodel.dap.base import Channel, Property
from marvin.utils.general import FuzzyDict, turn_off_ion

from .core import MarvinToolsClass
from .mixins import DAPallMixIn, GetApertureMixIn, NSAMixIn
from .quantities import AnalysisProperty


try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None


__all__ = ['Maps']


class Maps(MarvinToolsClass, NSAMixIn, DAPallMixIn, GetApertureMixIn):
    """A class that represents a DAP MAPS file.

    Provides access to the data stored in a DAP MAPS file. In addition to
    the parameters and variables defined for `~.MarvinToolsClass`, the
    following parameters and attributes are specific to `.Maps`.

    Parameters:
        bintype (str or None):
            The binning type. For MPL-4, one of the following: ``'NONE',
            'RADIAL', 'STON'`` (if ``None`` defaults to ``'NONE'``).
            For MPL-5, one of, ``'ALL', 'NRE', 'SPX', 'VOR10'``
            (defaults to ``'SPX'``). MPL-6 also accepts the ``'HYB10'`` binning
            schema.
        template (str or None):
            The stellar template used. For MPL-4, one of
            ``'M11-STELIB-ZSOL', 'MILES-THIN', 'MIUSCAT-THIN'`` (if ``None``,
            defaults to ``'MIUSCAT-THIN'``). For MPL-5 and successive, the only
            option in ``'GAU-MILESHC'`` (``None`` defaults to it).

    Attributes:
        header (`astropy.io.fits.Header`):
            The header of the datacube.
        wcs (`astropy.wcs.WCS`):
            The WCS solution for this plate

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None,
                 drpall=None, download=None, nsa_source='auto',
                 bintype=None, template=None, template_kin=None):

        if template_kin is not None:
            warnings.warn('template_kin is deprecated and will be removed in a future version.',
                          DeprecationWarning)
            template = template_kin if template is None else template

        # _set_datamodel will replace these strings with datamodel objects.
        self.bintype = bintype
        self.template = template

        self._bitmasks = None

        MarvinToolsClass.__init__(self, input=input, filename=filename,
                                  mangaid=mangaid, plateifu=plateifu,
                                  mode=mode, data=data, release=release,
                                  drpall=drpall, download=download)

        NSAMixIn.__init__(self, nsa_source=nsa_source)

        self.header = None
        self.wcs = None
        self._shape = None

        if self.data_origin == 'file':
            self._load_maps_from_file(data=self.data)
        elif self.data_origin == 'db':
            self._load_maps_from_db(data=self.data)
        elif self.data_origin == 'api':
            self._load_maps_from_api()
        else:
            raise marvin.core.exceptions.MarvinError(
                'data_origin={0} is not valid'.format(self.data_origin))

        self._check_versions(self)

    def __repr__(self):
        return ('<Marvin Maps (plateifu={0.plateifu!r}, mode={0.mode!r}, '
                'data_origin={0.data_origin!r}, bintype={0.bintype.name!r}, '
                'template={0.template.name!r})>'.format(self))

    def __getitem__(self, value):
        """Gets either a spaxel or a map depending on the type on input."""

        if isinstance(value, tuple):
            assert len(value) == 2, 'slice must have two elements.'
            y, x = value
            return self.getSpaxel(x=x, y=y, xyorig='lower')
        elif isinstance(value, six.string_types):
            return self.getMap(value)
        else:
            raise marvin.core.exceptions.MarvinError('invalid type for getitem.')

    def __getattr__(self, value):

        if isinstance(value, six.string_types) and value in self.datamodel:
            return self.getMap(value)

        return super(Maps, self).__getattribute__(value)

    def __dir__(self):

        class_members = list(list(zip(*inspect.getmembers(self.__class__)))[0])
        instance_attr = list(self.__dict__.keys())

        return sorted(class_members + instance_attr) + [prop.full() for prop in self.datamodel]

    def _set_datamodel(self):
        """Sets the datamodel."""

        self.datamodel = datamodel[self.release].properties
        self._bitmasks = datamodel[self.release].bitmasks
        self.bintype = self.datamodel.parent.get_bintype(self.bintype)
        self.template = self.datamodel.parent.get_template(self.template)

    def __deepcopy__(self, memo):
        return Maps(plateifu=copy.deepcopy(self.plateifu, memo),
                    release=copy.deepcopy(self.release, memo),
                    bintype=copy.deepcopy(self.bintype, memo),
                    template=copy.deepcopy(self.template, memo),
                    nsa_source=copy.deepcopy(self.nsa_source, memo))

    @staticmethod
    def _check_versions(instance):
        """Confirm that drpver and dapver match the ones from the header.

        This is written as a staticmethod because we'll also use if for
        ModelCube.

        """

        header_drpver = instance.header['VERSDRP3']

        isMPL4 = False

        if instance.release == 'MPL-4' and header_drpver == 'v1_5_0':
            header_drpver = 'v1_5_1'
            isMPL4 = True

        assert header_drpver == instance._drpver, ('mismatch between maps._drpver={0} '
                                                   'and header drpver={1}'
                                                   .format(instance._drpver, header_drpver))

        # MPL-4 does not have VERSDAP
        if isMPL4:
            assert 'VERSDAP' not in instance.header, \
                ('VERSDAP is present in the header but this is a MPL-4 MAPS. '
                 'That should not happen.')
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

        if self.release == 'MPL-4':
            niter = int('{0}{1}'.format(self.template.n, self.bintype.n))
            params = dict(drpver=self._drpver, dapver=self._dapver,
                          plate=plate, ifu=ifu, bintype=self.bintype.name,
                          n=niter, path_type='mangamap')
        else:
            daptype = '{0}-{1}'.format(self.bintype.name, self.template.name)
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

        self._check_file(self.header, self.data, 'Maps')

        # We use EMLINE_GFLUX because is present in MPL-4 and 5 and is not expected to go away.
        header = self.data['EMLINE_GFLUX'].header
        naxis = header['NAXIS']
        wcs_pre = astropy.wcs.WCS(header)

        # Takes only the first two axis.
        self.wcs = wcs_pre.sub(2) if naxis > 2 else naxis

        self._shape = (header['NAXIS2'], header['NAXIS1'])

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
        self.datamodel = datamodel[self._dapver].properties

        # Checks the bintype and template from the header
        is_MPL4 = 'MPL-4' in self.datamodel.parent.aliases
        if not is_MPL4:
            header_bintype = self.data[0].header['BINKEY'].strip().upper()
            header_bintype = 'SPX' if header_bintype == 'NONE' else header_bintype
        else:
            header_bintype = self.data[0].header['BINTYPE'].strip().upper()

        header_template_key = 'TPLKEY' if is_MPL4 else 'SCKEY'
        header_template = self.data[0].header[header_template_key].strip().upper()

        if self.bintype.name != header_bintype:
            self.bintype = self.datamodel.parent.get_bintype(header_bintype)

        if self.template.name != header_template:
            self.template = self.datamodel.parent.get_template(header_template)

    def _load_maps_from_db(self, data=None):
        """Loads the ``mangadap.File`` object for this Maps."""

        mdb = marvin.marvindb

        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise marvin.core.exceptions.MarvinError('No db connected')

        if sqlalchemy is None:
            raise marvin.core.exceptions.MarvinError('sqlalchemy required to access the local DB.')

        dm = datamodel[self.release]
        if dm.db_only:
            if self.bintype not in dm.db_only:
                raise marvin.core.exceptions.MarvinError('Specified bintype {0} is not available '
                                                         'in the DB'.format(self.bintype.name))

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
                                    dapdb.BinType.name == self.bintype.name,
                                    dapdb.Template.name == self.template.name).all()

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

        # Creates the WCS from the cube's WCS header
        self.wcs = astropy.wcs.WCS(self.data.cube.wcs.makeHeader())

        self._shape = self.data.cube.shape.shape

    def _load_maps_from_api(self):
        """Loads a Maps object from remote."""

        url = marvin.config.urlmap['api']['getMaps']['url']

        url_full = url.format(name=self.plateifu,
                              bintype=self.bintype.name,
                              template=self.template.name)

        try:
            response = self._toolInteraction(url_full)
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when checking if remote maps exists: {0}'.format(str(ee)))

        data = response.getData()

        if self.plateifu not in data['plateifu']:
            raise marvin.core.exceptions.MarvinError('remote maps has a different plateifu!')

        self.header = astropy.io.fits.Header.fromstring(data['header'])

        # Sets the mangaid
        self.mangaid = data['mangaid']

        # Sets the WCS
        self.wcs = astropy.wcs.WCS(astropy.io.fits.Header.fromstring(data['wcs']))

        self._shape = data['shape']

        return

    def _get_spaxel_quantities(self, x, y, spaxel=None):
        """Returns a dictionary of spaxel quantities."""

        maps_quantities = FuzzyDict({})

        if self.data_origin == 'file' or self.data_origin == 'db':

            # Stores a dictionary of (table, row)
            _db_rows = {}

            for dm in self.datamodel:

                data = {'value': None, 'ivar': None, 'mask': None}

                for key in data:

                    if key == 'ivar' and not dm.has_ivar():
                        continue
                    if key == 'mask' and not dm.has_mask():
                        continue

                    if self.data_origin == 'file':

                        extname = dm.name + '' if key == 'value' else dm.name + '_' + key

                        if dm.channel:
                            data[key] = self.data[extname].data[dm.channel.idx, y, x]
                        else:
                            data[key] = self.data[extname].data[y, x]

                    elif self.data_origin == 'db':

                        mdb = marvin.marvindb

                        table = getattr(mdb.dapdb, dm.model)

                        if table not in _db_rows:
                            _db_rows[table] = mdb.session.query(table).filter(
                                table.file_pk == self.data.pk, table.x == x, table.y == y).one()

                        colname = dm.db_column(ext=None if key == 'value' else key)
                        data[key] = getattr(_db_rows[table], colname)

                quantity = AnalysisProperty(data['value'], unit=dm.unit, ivar=data['ivar'],
                                            mask=data['mask'], pixmask_flag=dm.pixmask_flag)

                if spaxel:
                    quantity._init_bin(spaxel=spaxel, parent=self, datamodel=dm)

                maps_quantities[dm.full()] = quantity

        if self.data_origin == 'api':

            params = {'release': self._release}
            url = marvin.config.urlmap['api']['getMapsQuantitiesSpaxel']['url']

            try:
                response = self._toolInteraction(url.format(name=self.plateifu,
                                                            x=x, y=y,
                                                            bintype=self.bintype.name,
                                                            template=self.template.name,
                                                            params=params))
            except Exception as ee:
                raise marvin.core.exceptions.MarvinError(
                    'found a problem when checking if remote cube exists: {0}'.format(str(ee)))

            data = response.getData()

            for dm in self.datamodel:

                quantity = AnalysisProperty(data[dm.full()]['value'],
                                            ivar=data[dm.full()]['ivar'],
                                            mask=data[dm.full()]['mask'],
                                            unit=dm.unit,
                                            pixmask_flag=dm.pixmask_flag)

                if spaxel:
                    quantity._init_bin(spaxel=spaxel, parent=self, datamodel=dm)

                maps_quantities[dm.full()] = quantity

        return maps_quantities

    def get_binid(self, property=None):
        """Returns the binid map associated with a property.

        Parameters
        ----------
        property : `datamodel.Property` or None
            The property for which the associated binid map will be returned.
            If ``binid=None``, the default binid is returned.

        Returns
        -------
        binid : `Map`
            A `Map` with the binid associated with ``property`` or the default
            binid.

        """

        assert property is None or isinstance(property, Property), \
            'property must be None or a Property.'

        if property is None:
            assert self.datamodel.parent.default_binid is not None
            binid = self.datamodel.parent.default_binid
        else:
            binid = property.binid

        return self.getMap(binid)

    def getCube(self):
        """Returns the :class:`~marvin.tools.cube.Cube` for with this Maps."""

        if self.data_origin == 'db':
            cube_data = self.data.cube
        else:
            cube_data = None

        return marvin.tools.cube.Cube(data=cube_data,
                                      plateifu=self.plateifu,
                                      release=self.release)

    def getModelCube(self):
        """Returns the `~marvin.tools.cube.ModelCube` for with this Maps."""

        return marvin.tools.modelcube.ModelCube(plateifu=self.plateifu,
                                                release=self.release,
                                                bintype=self.bintype,
                                                template=self.template)

    def getSpaxel(self, x=None, y=None, ra=None, dec=None,
                  drp=True, models=False, model=None, **kwargs):
        """Returns the :class:`~marvin.tools.spaxel.Spaxel` matching certain coordinates.

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
            drp (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding DRP data.
            models (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding `.ModelCube` data.


        Returns:
            spaxels (list):
                The |spaxel|_ objects for this cube/maps corresponding to the
                input coordinates. The length of the list is equal to the
                number of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        if model is not None:
            raise marvin.core.exceptions.MarvinDeprecationError(
                'the model parameter has been deprecated. Use models.')

        return marvin.utils.general.general.getSpaxel(
            x=x, y=y, ra=ra, dec=dec,
            cube=drp, maps=self, modelcube=models, **kwargs)

    def _match_properties(self, property_name, channel=None, exact=False):
        """Returns the best match for a property_name+channel."""

        channel = channel.name if isinstance(channel, Channel) else channel

        channel = None if channel == 'None' else channel
        if channel is not None:
            property_name = property_name + '_' + channel

        best = self.datamodel[property_name]
        assert isinstance(best, Property), 'the retrived value is not a property.'

        if exact:
            assert best.full() == property_name, \
                'retrieved property {0!r} does not match input {1!r}'.format(best.full(),
                                                                             property_name)

        return best

    def getMap(self, property_name, channel=None, exact=False):
        """Retrieves a :class:`~marvin.tools.quantities.Map` object.

        Parameters:
            property_name (str):
                The property of the map to be extractred. It may the name
                of the channel (e.g. ``'emline_gflux_ha_6564'``) or just the
                name of the property (``'emline_gflux'``).
            channel (str or None):
                If defined, the name of the channel to be appended to
                ``property_name`` (e.g., ``'ha_6564'``).
            exact (bool):
                If ``exact=False``, fuzzy matching will be used, retrieving
                the best match for the property name and channel. If ``True``,
                will check that the name of returned map matched the input
                value exactly.

        """

        if isinstance(property_name, Property):
            best = property_name
        else:
            best = self._match_properties(property_name, channel=channel, exact=exact)

        # raise error when property is MPL-6 stellar_sigmacorr
        if best.full() == 'stellar_sigmacorr' and self.release == 'MPL-6':
            raise marvin.core.exceptions.MarvinError('stellar_sigmacorr is unreliable in MPL-6. '
                                                     'Please use MPL-7.')

        return marvin.tools.quantities.Map.from_maps(self, best)

    def getMapRatio(self, property_name, channel_1, channel_2):
        """Returns a ratio `~marvin.tools.quantities.Map`.

        .. attention::
            Deprecated, see :ref:`Enhanced Map<marvin-enhanced-map>`.

        For a given ``property_name``, returns a `~marvin.tools.quantities.Map`
        which is the ratio of ``channel_1/channel_2``.

        For a given ``property_name``, returns a `~marvin.tools.quantities.Map`
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

        return map_1 / map_2

    def is_binned(self):
        """Returns True if the Maps is not unbinned."""

        return self.bintype.binned

    def get_unbinned(self):
        """Returns a version of ``self`` corresponding to the unbinned Maps."""

        if self.is_binned is False:
            return self
        else:
            return Maps(plateifu=self.plateifu, release=self.release,
                        bintype=self.datamodel.parent.get_unbinned(),
                        template=self.template, mode=self.mode)

    def get_bpt(self, method='kewley06', snr_min=3, return_figure=True,
                show_plot=True, use_oi=True, **kwargs):
        """Returns the BPT diagram for this target.

        This method produces the BPT diagram for this target using emission
        line maps and returns a dictionary of classification masks, that can be
        used to select spaxels that have been classified as belonging to a
        certain excitation process. It also provides plotting functionalities.

        Extensive documentation can be found in :ref:`marvin-bpt`.

        Parameters:
            method ({'kewley06'}):
                The method used to determine the boundaries between different
                excitation mechanisms. Currently, the only available method is
                ``'kewley06'``, based on Kewley et al. (2006). Other methods
                may be added in the future. For a detailed explanation of the
                implementation of the method check the
                :ref:`BPT documentation <marvin-bpt>`.
            snr_min (float or dict):
                The signal-to-noise cutoff value for the emission lines used
                to generate the BPT diagram. If ``snr_min`` is a single value,
                that signal-to-noise will be used for all the lines.
                Alternatively, a dictionary of signal-to-noise values, with the
                emission line channels as keys, can be used. E.g.,
                ``snr_min={'ha': 5, 'nii': 3, 'oi': 1}``. If some values are
                not provided, they will default to ``SNR>=3``.
            return_figure (bool):
                If ``True``, it also returns the matplotlib
                `~matplotlib.figure.Figure` of the BPT diagram plot, which can
                be used to modify the style of the plot.
            show_plot (bool):
                If ``True``, interactively display the BPT plot.
            use_oi (bool):
                If ``True``, turns uses the OI diagnostic line in classifying
                BPT spaxels

        Returns:
            bpt_return:
                ``get_bpt`` always returns a dictionary of classification
                masks. These classification masks (not to be confused with
                bitmasks) are boolean arrays with the same shape as the
                `~marvin.tools.maps.Maps` or `~marvin.tools.cube.Cube` (without
                the spectral dimension) that can be used to select spaxels
                belonging to a certain excitation process (e.g., star forming).
                The keys of the dictionary, i.e., the classification
                categories, may change depending on the selected method.
                Consult the :ref:`BPT <marvin-bpt>` documentation for more
                details. If ``return_figure=True``, ``~.Maps.get_bpt`` will
                also return the matplotlib `~matplotlib.figure.Figure` for the
                generated plot, and a list of axes for each one of the
                subplots.

        Example:

            >>> cube = Cube(plateifu='8485-1901')
            >>> maps = cube.getMaps()
            >>> bpt_masks, bpt_figure = maps.get_bpt(snr=5, return_figure=True,
            >>>                                      show_plot=False)

            Now we can use the masks to select star forming spaxels from the
            cube

            >>> sf_spaxels = cube.flux[bpt_masks['sf']['global']]

            And we can save the figure as a PDF

            >>> bpt_figure.savefig('8485_1901_bpt.pdf')

        .. _figure: http://matplotlib.org/api/figure_api.html

        """

        if 'snr' in kwargs:
            warnings.warn('snr is deprecated. Use snr_min instead. '
                          'snr will be removed in a future version of marvin',
                          marvin.core.exceptions.MarvinDeprecationWarning)
            snr_min = kwargs.pop('snr')

        if len(kwargs.keys()) > 0:
            raise marvin.core.exceptions.MarvinError(
                'unknown keyword {0}'.format(list(kwargs.keys())[0]))

        # Makes sure all the keys in the snr keyword are lowercase
        if isinstance(snr_min, dict):
            snr_min = dict((kk.lower(), vv) for kk, vv in snr_min.items())

        # If we don't want the figure but want to show the plot, we still need to
        # temporarily get it.
        do_return_figure = True if return_figure or show_plot else False

        with turn_off_ion(show_plot=show_plot):
            bpt_return = marvin.utils.dap.bpt.bpt_kewley06(self, snr_min=snr_min,
                                                           return_figure=do_return_figure,
                                                           use_oi=use_oi)

        # Returs what we actually asked for.
        if return_figure and do_return_figure:
            return bpt_return
        elif not return_figure and do_return_figure:
            return bpt_return[0]
        else:
            return bpt_return

    def to_dataframe(self, columns=None, mask=None):
        """Converts the maps object into a Pandas dataframe.

        Parameters:
            columns (list):
                The properties+channels you want to include.
                Defaults to all of them.
            mask (array):
                A 2D mask array for filtering your data output

        Returns:
            df (`~pandas.DataFrame`):
                A Pandas `~pandas.DataFrame`.

        """

        allprops = [p.full() for p in self.datamodel]

        if columns:
            allprops = [p for p in allprops if p in columns]
        data = np.array([self[p].value[mask].flatten() for p in allprops])

        # add a column for spaxel index
        spaxarr = np.array([np.where(mask.flatten())[0]]) \
            if mask is not None else np.array([np.arange(data.shape[1])])
        data = np.concatenate((spaxarr, data), axis=0)
        allprops = ['spaxelid'] + allprops

        # create the dataframe
        df = pd.DataFrame(data.transpose(), columns=allprops)
        return df
