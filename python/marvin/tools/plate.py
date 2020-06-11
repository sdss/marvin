#!/usr/bin/env python
# encoding: utf-8

# Created by Brian Cherinka on 2016-05-17 10:17:35
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-05-17 10:17:35 by Brian Cherinka
#     Last Modified On: 2016-05-17 10:17:35 by Brian

from __future__ import division, print_function

import inspect

from astropy.io import fits

from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin.utils.general.structs import FuzzyList

from .core import MarvinToolsClass


try:
    from sdss_access.path import Path
except ImportError:
    Path = None


class Plate(MarvinToolsClass, FuzzyList):
    '''A class to interface with MaNGA Plate.

    This class represents a Plate, initialised either
    from a file, a database, or remotely via the Marvin API. The class
    inherits from Python's list class, and is defined as a list of
    Cube objects.  As it inherits from list, it can do all the standard Python
    list operations.

    When instanstantiated, Marvin Plate will attempt to discover and load all the Cubes
    associated with this plate.

    Parameters:
        plate (str):
            The plate id of the Plate to load.
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

    Attributes:
        cubeXXXX (object):
            The Marvin Cube object for the given ifu, e.g. cube1901 refers to the Cube for plateifu 8485-1901
        plate/plateid (int):
            The plate id for this plate
        cartid (str):
            The cart id for this plate
        designid (int):
            The design id for this plate
        ra (float):
            The RA of the plate center
        dec (float):
            The declination of the plate center
        dateobs (str):
            The date of observation for this plate
        surveymode (str):
            The survey mode for this plate
        isbright (bool):
            True if this is a bright time plate

    Return:
        plate:
            An object representing the Plate entity. The object is a list of
            Cube objects, one for each IFU cube in the Plate entity.

    Example:
        >>> from marvin.tools.plate import Plate
        >>> plate = Plate(plate=8485)
        >>> print(plate)
        >>> <Marvin Plate (plate=8485, n_cubes=17, mode='local', data_origin='db')>
        >>>
        >>> print('Cubes found in this plate: {0}'.format(len(plate)))
        >>> Cubes found in this plate: 4
        >>>
        >>> # access the plate via index to access the individual cubes
        >>> plate[0]
        >>> <Marvin Cube (plateifu='8485-12701', mode='local', data_origin='db')>
        >>>
        >>> # or by name
        >>> plate['12702']
        >>> <Marvin Cube (plateifu='8485-12702', mode='local', data_origin='db')>
        >>>
    '''

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None, plate=None,
                 download=None, nocubes=None):

        self._cubes = None
        self._plate = None
        self._pdict = None
        self.platedir = None
        self.nocubes = nocubes

        # If plateid specified, force a temp plateifu
        if plate:
            self.plateid = plate
            plateifu = '{0}-XXXX'.format(self.plateid)

        self.plateifu = plateifu

        args = [plate, plateifu]
        assert any(args), 'Enter plate or plateifu!'

        MarvinToolsClass.__init__(self, input=input, filename=filename, mangaid=mangaid,
                                  plateifu=plateifu, mode=mode, data=data, release=release,
                                  download=download)

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

        return ('<Marvin Plate (plate={self.plateid!r}, n_cubes={0}, mode={self.mode!r}, '
                'data_origin={self.data_origin!r})>'.format(len(self), self=self))

    def __dir__(self):
        ''' Overriding dir for Plate '''

        # get the attributes from the class itself
        class_members = list(list(zip(*inspect.getmembers(self.__class__)))[0])
        instance_attr = list(self.__dict__.keys())
        # get the dir from FuzzyList
        listattr = ['cube{0}'.format(i.plateifu.split('-')[1]) for i in self]
        listattr.sort()

        return listattr + sorted(class_members + instance_attr)

    def __getattr__(self, value):

        if 'cube' in value:
            ifu = value.split('cube')[-1]
            plateifu = '{0}-{1}'.format(self.plate, ifu)
            return self[plateifu]

        return super(Plate, self).__getattribute__(value)

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
        from marvin import marvindb as mdb

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
        response = self._toolInteraction(url)
        data = response.getData()
        self._hdr = data['header']
        self.data_origin = 'api'
        self._makePdict()

    def _initCubes(self):
        ''' Initialize a list of Marvin Cube objects '''

        _cubes = [None]
        if self.data_origin == 'file':
            sdss_path = Path(release=self.release)
            if self.dir3d == 'stack':
                cubes = sdss_path.expand('mangacube', drpver=self._drpver,
                                         plate=self.plateid, ifu='*')
            else:
                cubes = sdss_path.expand('mangamastar', drpver=self._drpver,
                                         plate=self.plateid, ifu='*')
            _cubes = [Cube(filename=cube, mode=self.mode, release=self.release) for cube in cubes]

        elif self.data_origin == 'db':
            _cubes = [Cube(plateifu=cube.plateifu, mode=self.mode, release=self.release)
                      for cube in self._plate.cubes]

        elif self.data_origin == 'api':
            routeparams = {'plateid': self.plateid}
            url = config.urlmap['api']['getPlateCubes']['url'].format(**routeparams)

            # Make the API call
            response = self._toolInteraction(url)
            data = response.getData()
            plateifus = data['plateifus']
            _cubes = [Cube(plateifu=pifu, mode=self.mode, release=self.release) for pifu in plateifus]

        FuzzyList.__init__(self, _cubes)
        self.mapper = (lambda e: e.plateifu)

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

    def _checkFilename(self):
        ''' Checks the filename for a proper FITS file '''

        # if filename is not FITS, then try to load one
        if 'fits' not in self.filename.lower():

            if not Path:
                raise MarvinError('sdss_access is not installed')
            else:
                # is_public = 'DR' in self._release
                # path_release = self._release.lower() if is_public else None
                sdss_path = Path(release=self._release)

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
