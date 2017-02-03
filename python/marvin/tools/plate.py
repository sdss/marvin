#!/usr/bin/env python
# encoding: utf-8

# Created by Brian Cherinka on 2016-05-17 10:17:35
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-05-17 10:17:35 by Brian Cherinka
#     Last Modified On: 2016-05-17 10:17:35 by Brian

from __future__ import print_function
from __future__ import division
from marvin.core.core import MarvinToolsClass
from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin import config, marvindb
from astropy.io import fits
from brain.utils.general import checkPath

try:
    from sdss_access.path import Path
except ImportError:
    Path = None


class Plate(MarvinToolsClass, list):
    '''A class to interface with MaNGA Plate.

    This class represents a Plate, initialised either
    from a file, a database, or remotely via the Marvin API. The class
    inherits from Python's list class, and is defined as a list of
    Cube objects.  As it inherits from list, it can do all the standard Python
    list operations.

    When instanstantiated, Marvin Plate will attempt to discover and load all the Cubes
    associated with this plate.

    Parameters:
        plateid (str):
            The plateid of the Plate to load.
        plateifu (str):
            The plate-ifu of the Plate to load
        filename (str):
            The path of the file containing the data cube to load.
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`..
        release (str):
            The MPL/DR version of the data to use.
        nocubes (bool):
            Set this to turn off the Cube loading
    Return:
        plate:
            An object representing the Plate entity. The object is a list of
            Cube objects, one for each IFU cube in the Plate entity.

    Example:
        >>> from marvin.tools.plate import Plate
        >>> plate = Plate(plateid=8485)
        >>> print(plate)
        >>> <Marvin Plate (plateid=8485, mode='local', data_origin='db')>
        >>>
        >>> print('Cubes found in this plate: {0}'.format(len(plate)))
        >>> Cubes found in this plate: 4
        >>>
        >>> # access the plate via index to access the individual cubes
        >>> print(plate[0])
        >>> <Marvin Cube (plateifu='8485-12701', mode='local', data_origin='db')>
        >>>
        >>> print(plate[1])
        >>> <Marvin Cube (plateifu='8485-12702', mode='local', data_origin='db')>
        >>>
    '''

    def __init__(self, *args, **kwargs):

        self.plateid = kwargs.get('plateid', None)
        self._cubes = None
        self._plate = None
        self._pdict = None
        self.platedir = None
        self.nocubes = kwargs.get('nocubes', None)

        # If plateid specified, force a temp plateifu
        if self.plateid:
            self.plateifu = '{0}-XXXX'.format(self.plateid)
            kwargs['plateifu'] = self.plateifu

        self.plateifu = kwargs.get('plateifu', None)

        args = [self.plateid, self.plateifu]
        assert any(args), 'Enter plateid or plateifu!'

        MarvinToolsClass.__init__(self, *args, **kwargs)

        # sort out any plateid, plate-ifu, mangaid name snafus
        self._sortOutNames()

        # grab the plate info
        if self.data_origin == 'file':
            self._getPlateFromFile()
        elif self.data_origin == 'db':
            self._getPlateFromDB()
        elif self.data_origin == 'api':
            self._getPlateFromAPI()

        # load the plate params and init the Marvin Cubes
        self._setParams()
        if not self.nocubes:
            self._initCubes()

    def __repr__(self):
        '''Representation for Plate.'''

        return ('<Marvin Plate (plateid={self.plateid!r}, mode={self.mode!r}, '
                'data_origin={self.data_origin!r})>'.format(self=self))

    def _getFullPath(self, **kwargs):
        """Returns the full path of the file in the tree."""
        self.filename = super(Plate, self)._getFullPath('mangaplate', drpver=self._drpver,
                                                        plate=self.plateid, **kwargs)
        self.platedir = self.filename
        self._checkFilename()
        return self.filename

    def _getPlateFromFile(self):
        ''' Initialize a Plate from a Cube/RSS File'''

        # Load file
        try:
            self._hdr = fits.getheader(self.filename, 1)
            self.plateid = int(self._hdr['PLATEID'])
        except Exception as e:
            raise MarvinError('Could not initialize via filename: {0}'
                              .format(e))
        else:
            self.data_origin = 'file'
            self._makePdict()

    def _getPlateFromDB(self):
        ''' Initialize a Plate from the DB '''
        import sqlalchemy

        mdb = marvindb

        if not mdb.isdbconnected:
            raise MarvinError('No db connected')

        # Grab any cube for this plate
        cube = None
        try:
            cube = mdb.session.query(mdb.datadb.Cube).join(
                mdb.datadb.PipelineInfo, mdb.datadb.PipelineVersion).\
                filter(mdb.datadb.Cube.plate == self.plateid,
                       mdb.datadb.PipelineVersion.version == self._drpver).first()
        except sqlalchemy.orm.exc.NoResultFound as ee:
            raise MarvinError('Could not retrieve Cube for plate {0}: '
                              'No Results Found: {1}'
                              .format(self.plateid, ee))

        except Exception as ee:
            raise MarvinError('Could not retrieve Cube for plate {0}: '
                              'Unknown exception: {1}'
                              .format(self.plateid, ee))
        else:
            # no cube
            if not cube:
                raise MarvinError('No cube found in db for plate {0}, drpver {1}'
                                  .format(self.plateid, self._drpver))
            # cube but no plateclass
            try:
                self._plate = cube.plateclass
            except AttributeError as ee:
                raise MarvinError('AttributeError: cube has no plateclass for plate {0}: {1}'
                                  .format(self.plateid, ee))
            else:
                self._hdr = self._plate._hdr
                self._pdict = self._plate.__dict__
                self.data_origin = 'db'

        if not self._plate:
            raise MarvinError('Could not retrieve Plate for id {0}'.format(self.plateid))

    def _getPlateFromAPI(self):
        ''' Initialize a Plate using the API '''

        # Checks that the Plate exists.
        routeparams = {'plateid': self.plateid}
        url = config.urlmap['api']['getPlate']['url'].format(**routeparams)

        # Make the API call
        response = self.ToolInteraction(url)
        data = response.getData()
        self._hdr = data['header']
        self.data_origin = 'api'
        self._makePdict()

    @checkPath
    def _initCubes(self):
        ''' Initialize a list of Marvin Cube objects '''

        _cubes = [None]
        if self.data_origin == 'file':
            sdss_path = Path()
            if self.dir3d == 'stack':
                cubes = sdss_path.expand('mangacube', drpver=self._drpver,
                                         plate=self.plateid, ifu='*')
            else:
                cubes = sdss_path.expand('mangamastar', drpver=self._drpver,
                                         plate=self.plateid, ifu='*')
            _cubes = [Cube(filename=cube) for cube in cubes]

        elif self.data_origin == 'db':
            _cubes = [Cube(plateifu=cube.plateifu)
                      for cube in self._plate.cubes]

        elif self.data_origin == 'api':
            routeparams = {'plateid': self.plateid}
            url = config.urlmap['api']['getPlateCubes']['url'].format(**routeparams)

            # Make the API call
            response = self.ToolInteraction(url)
            data = response.getData()
            plateifus = data['plateifus']
            _cubes = [Cube(plateifu=pifu, mode='remote') for pifu in plateifus]

        list.__init__(self, _cubes)

    def _setParams(self):
        ''' Set the plate parameters '''
        self.ra = self._pdict.get('ra', None)
        self.dec = self._pdict.get('dec', None)
        self.designid = self._pdict.get('designid', None)
        self.cartid = self._pdict.get('cartid', None)
        self.dateobs = self._pdict.get('dateobs', None)
        self.platetype = self._pdict.get('platetype', None)
        self.surveymode = self._pdict.get('surveymode', None)
        self.isbright = self._pdict.get('isbright', None)
        self.dir3d = self._pdict.get('dir3d', None)
        self.plateid = int(self.plateid)

    def _makePdict(self):
        ''' Make the necessary plate dictionary '''
        self._pdict = {}
        self._pdict['ra'] = self._hdr.get('CENRA', None)
        self._pdict['dec'] = self._hdr.get('CENDEC', None)
        self._pdict['designid'] = self._hdr.get('DESIGNID', None)
        self._pdict['cartid'] = self._hdr.get('CARTID', None)
        self._pdict['dateobs'] = self._hdr.get('DATE-OBS', None)
        self._pdict['platetype'] = self._hdr.get('PLATETYP', None)
        self._pdict['surveymode'] = self._hdr.get('SRVYMODE', None)
        self._pdict['isbright'] = 'APOGEE' in self._pdict['surveymode']
        self._pdict['dir3d'] = 'mastar' if self._pdict['isbright'] else 'stack'

        self._pdict['ra'] = float(self._pdict['ra'])
        self._pdict['dec'] = float(self._pdict['dec'])
        self._pdict['designid'] = float(self._pdict['designid'])

    def _sortOutNames(self):
        ''' Sort out any name issues with plateid, plateifu, mangaid inputs '''

        if self.plateifu and 'XXX' not in self.plateifu:
            plate, ifu = self.plateifu.split('-')
            self.plateid = int(plate)

    @checkPath
    def _checkFilename(self):
        ''' Checks the filename for a proper FITS file '''

        # if filename is not FITS, then try to load one
        if 'fits' not in self.filename.lower():
            sdss_path = Path()
            # try a cube
            full = sdss_path.full('mangacube', drpver=self._drpver, plate=self.plateid, ifu='*')
            cubeexists = sdss_path.any('', full=full)
            if cubeexists:
                file = sdss_path.one('', full=full)
            else:
                # try an rss
                full = sdss_path.full('mangarss', drpver=self._drpver, plate=self.plateid, ifu='*')
                rssexists = sdss_path.any('', full=full)
                if rssexists:
                    file = sdss_path.one('', full=full)
                else:
                    file = None
            # load the file
            if file:
                self.filename = file
            else:
                self.filename = None
