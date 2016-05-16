import os
import re
import warnings
from marvin.core.exceptions import MarvinUserWarning, MarvinError
from brain.utils.general.general import getDbMachine
from collections import OrderedDict
from brain import bconfig

# Inits the log
from brain.core.logger import initLog

# Defines log dir.
if 'MARVIN_LOGS_DIR' in os.environ:
    logFilePath = os.path.join(os.path.realpath(os.environ['MARVIN_LOGS_DIR']), 'marvin.log')
else:
    logFilePath = os.path.realpath(os.path.join(os.environ['HOME'], '.marvin', 'marvin.log'))

log = initLog(logFilePath)

warnings.simplefilter('once')


class MarvinConfig(object):

    def __init__(self):

        self._drpall = None
        self._inapp = False

        self.drpver = None
        self.dapver = None
        self.drver = None
        self.mplver = None
        self.vermode = None
        self.download = False
        self._setDbConfig()
        self._checkConfig()

        self.setDefaultDrpAll()

    def setDefaultDrpAll(self, drpver=None):
        """Tries to set the default location of drpall."""

        if not drpver and not self.drpver:
            return

        self.drpall = self._getDrpAllPath(drpver=drpver)

    def _getDrpAllPath(self, drpver=None):
        """Returns the default path for drpall, give a certain `drpver`."""

        drpver = drpver if drpver else self.drpver

        if 'MANGA_SPECTRO_REDUX' in os.environ and drpver:
            return os.path.join(os.environ['MANGA_SPECTRO_REDUX'], str(drpver),
                                'drpall-{0}.fits'.format(drpver))
        else:
            return None

############ Brain Config overrides ############
# These are configuration parameter defined in Brain.bconfig. We need
# to be able to modify them during run time, so we define properties and
# setters to do that from Marvin.config.

    @property
    def mode(self):
        return bconfig.mode

    @mode.setter
    def mode(self, value):
        bconfig.mode = value

    @property
    def sasurl(self):
        return bconfig.sasurl

    @sasurl.setter
    def sasurl(self, value):
        bconfig.sasurl = value

#################################################

    @property
    def drpall(self):
        return self._drpall

    @drpall.setter
    def drpall(self, value):
        if os.path.exists(value):
            self._drpall = value
        else:
            warnings.warn('path {0} cannot be found. Setting drpall to None.'
                          .format(value), MarvinUserWarning)

    def _setDbConfig(self):
        ''' Set the db configuration '''
        self.db = getDbMachine()

    def _checkConfig(self):
        ''' Check the config '''
        # set and sort the base MPL dictionary
        mpldict = {'MPL-4': ('v1_5_1', '1.1.1'), 'MPL-3': ('v1_3_3', 'v1_0_0'), 'MPL-2': ('v1_2_0', None),
                   'MPL-1': ('v1_0_0', None), 'DR13': ('v1_5_4', None)}
        mplsorted = sorted(mpldict.items(), key=lambda p: p[1][0], reverse=True)
        self._mpldict = OrderedDict(mplsorted)

        # Check the versioning config
        if not self.mplver or not (self.drpver and self.dapver):
            log.info('No MPL or DRP/DAP version set. Setting default to MPL-4')
            self.setMPL('MPL-4')

    def setMPL(self, mplver):
        ''' Set the data version by MPL '''

        m = re.search('MPL-([0-9])', mplver)
        assert m is not None, 'MPL version must be of form "MPL-[X]"'
        if m:
            self.mplver = mplver
            self.drpver, self.dapver = self.lookUpVersions(mplver)

    def setDR(self, drver):
        ''' Set the data version by Data Release '''
        m = re.search('DR1([3-9])', drver)
        assert m is not None, 'DR version must be of form "DR[XX]"'
        if m:
            self.drver = drver
            self.drpver, self.dapver = self.lookUpVersions(drver)

    def setVersions(self, drpver=None, dapver=None):
        ''' Set the data version by DRPVER and DAPVER '''

        if drpver:
            assert type(drpver) == str, 'drpver needs to be a string'
            drpre = re.search('v([0-9][_]([0-9])[_]([0-9]))', drpver)
            assert drpre is not None, 'DRP version must be of form "v[X]_[X]_[X]"'
            if drpre:
                self.drpver = drpver
                self.mplver = self.lookUpMpl(drpver)

        if dapver:
            assert type(dapver) == str, 'dapver needs to be a string'
            dapre1 = re.search('v([0-9][_]([0-9])[_]([0-9]))', dapver)
            dapre2 = re.search('([0-9][.]([0-9])[.]([0-9]))', dapver)
            assert (dapre1 or dapre2) is not None, \
                'DAP version must be of form "v[X]_[X]_[X]", or [X].[X].[X]'
            if any([dapre1, dapre2]):
                self.dapver = dapver

    def lookUpVersions(self, mplver):
        ''' Retrieve the DRP and DAP versions that make up an MPL '''

        try:
            drpver, dapver = self._mpldict[mplver]
        except KeyError as e:
            raise MarvinError('MPL version {0} not found in lookup table. No associated DRP/DAP versions. Should they be added?  Check for typos.'.format(mplver))

        return drpver, dapver

    def lookUpMpl(self, drpver):
        ''' Retrieve the MPL version for a given DRP version'''

        # Flip the mpldict
        verdict = {val[0]: key for key, val in self._mpldict.items()}

        try:
            mplver = verdict[drpver]
        except KeyError as e:
            raise MarvinError('DRP version {0} not found in lookup table. No associated MPL version. Should one be added?  Check for typos.'.format(drpver))

        return mplver

config = MarvinConfig()
config._checkConfig()

# Inits the Database session and ModelClasses
from marvin.db.marvindb import MarvinDB
marvindb = MarvinDB(dbtype=config.db)

# Inits the URL Route Map
from marvin.api.api import Interaction
# config.sasurl = 'http://cd057661.ngrok.io/'  # this is a temporary measure REMOVE THIS
# config.sasurl = 'http://8a7373c3.ngrok.io'  # Jose's ngrok


from brain.core.core import URLMapDict

try:
    response = Interaction('api/general/getroutemap', request_type='get')
except Exception as e:
    config.urlmap = URLMapDict()
else:
    config.urlmap = response.getRouteMap()
