#!/usr/bin/env python
# encoding: utf-8

# Created by Brian Cherinka on 2016-05-17 10:17:35
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-05-17 10:17:35 by Brian Cherinka
#     Last Modified On: 2016-05-17 10:17:35 by Brian

from __future__ import print_function
from __future__ import division
from marvin.core import MarvinToolsClass
from marvin.core import MarvinError
from marvin.tools.cube import Cube
from marvin import config, marvindb
import numpy as np
from astropy.io import fits


class Plate(MarvinToolsClass, list):
    ''' A class representing a Plate

    '''

    def __init__(self, *args, **kwargs):

        self.plateid = kwargs.get('plateid', None)
        self._cubes = None
        self._plate = None
        self._pdict = None
        self.nocubes = kwargs.get('nocubes', None)

        # If plateid specified, force a temp plateifu
        if self.plateid:
            self.plateifu = '{0}-XXXX'.format(self.plateid)
            kwargs['plateifu'] = self.plateifu

        MarvinToolsClass.__init__(self, *args, **kwargs)

        # sort out any plateid, plate-ifu, mangaid name snafus
        self._sortOutNames()

        # grab the plate info
        if self.data_origin == 'file':
            self._getPlateFromFile()
        elif self.data_origin == 'db':
            self._getPlateFromDB()
        elif self.data_origin == 'api':
            pass  # do api stuff

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
        return super(Plate, self)._getFullPath('mangaplate', drpver=self._drpver, plate=self.plateid, **kwargs)

    def _getPlateFromFile(self):
        ''' Initialize a Plate from a Cube/RSS File'''
        try:
            self._hdr = fits.getheader(self.filename)
            self.plateid = int(self._hdr['PLATEID'])
        except Exception as e:
            raise MarvinError('Could not initialize via filename: {0}'
                              .format(e))
        else:
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
        except sqlalchemy.orm.exc.NoResultFound as e:
            raise MarvinError('Could not retrieve Cube for plate {0}: '
                              'No Results Found: {1}'
                              .format(self.plateid, ee))

        except Exception as e:
            raise MarvinError('Could not retrieve Cube for plate {0}: '
                              'Unknown exception: {1}'
                              .format(self.plateid, ee))
        else:
            self._plate = cube.plateclass
            self._hdr = self._plate._hdr
            self._pdict = self._plate.__dict__

        if not self._plate:
            raise MarvinError('Could not retrieve Plate for id {0}'.format(self.plateid))

    def _initCubes(self):
        ''' '''

        _cubes = [None]
        if self.data_origin == 'file':
            # TODO - replace this will full sdss_access local implementation
            # but for now - use this temporary hack
            # also - Cube instantiation by filename is broken, causes segfault
            import glob
            import os
            import re
            cubes = glob.glob(os.path.join(self._getFullPath(), self.dir3d, '*LOGCUBE*'))
            plateifus = [re.search('(\d{4}[-]\d{3,5})', cube).group(0) for cube in cubes]
            _cubes = [Cube(plateifu=pifu) for pifu in plateifus]

        elif self.data_origin == 'db':
            _cubes = [Cube(plateifu=cube.plateifu)
                      for cube in self._plate.cubes]

        elif self.data_origin == 'api':
            pass

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

    def _sortOutNames(self):
        ''' Sort out any name issues with plateid, plateifu, mangaid inputs '''

        if self.plateifu and 'XXX' not in self.plateifu:
            plate, ifu = self.plateifu.split('-')
            self.plateid = int(plate)


