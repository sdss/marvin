import os
import re
import warnings
from marvin.core.exceptions import MarvinUserWarning, MarvinError
from brain.utils.general.general import getDbMachine
from collections import OrderedDict
from brain import bconfig
from brain.core.core import URLMapDict

# Inits the log
from brain.core.logger import initLog

# Defines log dir.
if 'MARVIN_LOGS_DIR' in os.environ:
    logFilePath = os.path.join(os.path.realpath(os.environ['MARVIN_LOGS_DIR']), 'marvin.log')
else:
    logFilePath = os.path.realpath(os.path.join(os.environ['HOME'], '.marvin', 'marvin.log'))

log = initLog(logFilePath)

warnings.simplefilter('once')
warnings.filterwarnings('ignore', 'Skipped unsupported reflection of expression-based index')


class MarvinConfig(object):
    ''' Global Marvin Configuration

    The global configuration of Marvin.

    Parameters:
        mplver (str):
            The MPL version of the MaNGA data you want to use
        download (bool):
            Set to turn on downloading of objects with sdss_access
    '''
    def __init__(self):

        self._check_sas_dir()

        self._drpall = None
        self._inapp = False

        self._urlmap = None
        self._xyorig = None

        self._mplver = None
        self._drver = None

        self.vermode = None
        self.download = False
        self._setDbConfig()
        self._checkConfig()

        self.setDefaultDrpAll()

    def _check_sas_dir(self):
        """Check if $SAS_BASE_DIR is defined. If it is not, creates and defines it."""

        if 'MANGA_SPECTRO_REDUX' in os.environ:
            return

        if 'SAS_BASE_DIR' not in os.environ:
            sas_base_dir = os.path.expanduser('~/sas')
            if not os.path.exists(sas_base_dir):
                warnings.warn('no SAS_BASE_DIR found. Creating it in {0}.'.format(sas_base_dir),
                              MarvinUserWarning)
                os.makedirs(sas_base_dir)
            os.environ['SAS_BASE_DIR'] = sas_base_dir

        if 'MANGA_SPECTRO_REDUX' not in os.environ:
            manga_spectro_redux = os.path.join(
                os.path.abspath(os.environ['SAS_BASE_DIR']), 'mangawork/manga/spectro/redux/')
            if not os.path.exists(manga_spectro_redux):
                warnings.warn('no MANGA_SPECTRO_REDUX found. Creating it in {0}.'
                              .format(manga_spectro_redux),  MarvinUserWarning)

                os.makedirs(manga_spectro_redux)
            os.environ['MANGA_SPECTRO_REDUX'] = manga_spectro_redux

    def setDefaultDrpAll(self, drpver=None):
        """Tries to set the default location of drpall."""

        if not drpver:
            drpver, __ = self.lookUpVersions(drver=self.drver, mplver=self.mplver)
        self.drpall = self._getDrpAllPath(drpver)

    def _getDrpAllPath(self, drpver):
        """Returns the default path for drpall, give a certain ``drpver``."""

        if 'MANGA_SPECTRO_REDUX' in os.environ and drpver:
            return os.path.join(os.environ['MANGA_SPECTRO_REDUX'], str(drpver),
                                'drpall-{0}.fits'.format(drpver))
        else:
            raise MarvinError('Must have the MANGA_SPECTRO_REDUX environment variable set')

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

    @property
    def mplver(self):
        return self._mplver

    @mplver.setter
    def mplver(self):
        raise MarvinError('mplver cannot be set directly. Use marvin.config.setMPL()')

    @property
    def drver(self):
        return self._drver

    @drver.setter
    def drver(self, value):
        raise MarvinError('drver cannot be set directly. Use marvin.config.setDR()')


#################################################

    @property
    def urlmap(self):
        """Retrieves the URLMap the first time it is needed."""

        if self._urlmap is None:
            try:
                response = Interaction('api/general/getroutemap', request_type='get')
            except Exception as e:
                warnings.warn('cannot retrieve URLMap. Remote functionality will not work.',
                              MarvinUserWarning)
                self._urlmap = URLMapDict()
            else:
                self._urlmap = response.getRouteMap()

        return self._urlmap

    @urlmap.setter
    def urlmap(self, value):
        """Manually sets the URLMap."""
        self._urlmap = value

    @property
    def xyorig(self):
        if not self._xyorig:
            self._xyorig = 'center'

        return self._xyorig

    @xyorig.setter
    def xyorig(self, value):

        assert value.lower() in ['center', 'lower'], 'xyorig must be center or lower.'

        self._xyorig = value.lower()

    @property
    def drpall(self):
        return self._drpall

    @drpall.setter
    def drpall(self, value):
        if os.path.exists(value):
            self._drpall = value
        else:
            self._drpall = None
            warnings.warn('path {0} cannot be found. Setting drpall to None.'
                          .format(value), MarvinUserWarning)

    def _setDbConfig(self):
        ''' Set the db configuration '''
        self.db = getDbMachine()

    def _checkConfig(self):
        ''' Check the config '''
        # set and sort the base MPL dictionary
        mpldict = {'MPL-5': ('v2_0_1', '2.0.2'), 'MPL-4': ('v1_5_1', '1.1.1'), 'MPL-3': ('v1_3_3', 'v1_0_0'),
                   'MPL-2': ('v1_2_0', None), 'MPL-1': ('v1_0_0', None)}  #, 'DR13': ('v1_5_4', None)}
        mplsorted = sorted(mpldict.items(), key=lambda p: p[1][0], reverse=True)
        self._mpldict = OrderedDict(mplsorted)

        # Check the versioning config
        if not self.mplver and not self.drver:
            topkey = self._mpldict.keys()[0]
            log.info('No MPL or DR version set. Setting default to {0}'.format(topkey))
            if 'DR' in topkey:
                self.setDR(topkey)
            else:
                self.setMPL(topkey)

    def setMPL(self, mplver):
        ''' Set the data version by MPL

        Sets the MPL version globally in the config. When specifying the MPL,
        the DRP and DAP versions also get set globally.

        Parameters:
            mplver (str):
                The MPL version to set, in form of MPL-X

        Example:
            >>> config.setMPL('MPL-4')
        '''

        mm = re.search('MPL-([0-9])', mplver)
        assert mm is not None, 'MPL version must be of form "MPL-[X]"'
        if mm:
            self._mplver = mplver
            self._drver = None

            drpver = self._mpldict[self._mplver][0]
            self.drpall = self._getDrpAllPath(drpver)

    def setDR(self, drver):
        ''' Set the data version by Data Release (DR)

        Sets the DR version globally in the config. When specifying the DR,
        the DRP and DAP versions also get set globally.

        Parameters:
            drver (str):
                The DR version to set, in form of DRXX

        Example:
            >>> config.setDR('DR13')
        '''
        mm = re.search('DR1([3-9])', drver)
        assert mm is not None, 'DR version must be of form "DR[XX]"'
        if mm:
            self._mplver = None
            self._drver = drver

            drpver = self._mpldict[self._drver][0]
            self.drpall = self._getDrpAllPath(drpver)

    def lookUpVersions(self, mplver=None, drver=None):
        """Retrieve the DRP and DAP versions that make up an MPL/DR

        Look up the DRP and DAP version for a specified MPL version.

        Parameters:
            mplver (str):
                The MPL version
            drver (str):
                The DR version

        Returns:
            drpver (str):
                The DRP version according to the input MPL version
            dapver (str):
                The DAP version according to the input MPL version

        """

        # Asserts one and only one of mplver or drver are set
        assert bool(mplver) != bool(drver), 'one and only one of drver or mplver must be set.'

        try:
            drpver, dapver = self._mpldict[mplver or drver]
        except KeyError as ee:
            raise MarvinError('MPL/DR version {0} not found in lookup table. '
                              'No associated DRP/DAP versions. '
                              'Should they be added?  Check for typos.'.format(mplver or drver))

        return drpver, dapver

    def lookUpMpl(self, drpver):
        ''' Retrieve the MPL version for a given DRP version

        Look up the MPL version for a specified DRP version.

        Parameters:
            drpver (str):
                The DRP version to use
        Returns:
            mplver (str):
                The MPL version according to the input DRP version
        '''

        # Flip the mpldict
        verdict = {val[0]: key for key, val in self._mpldict.items()}

        try:
            mplver = verdict[drpver]
        except KeyError as e:
            raise MarvinError('DRP version {0} not found in lookup table. No associated MPL version. Should one be added?  Check for typos.'.format(drpver))

        return mplver

    def switchSasUrl(self, sasmode='utah', ngrokid=None):
        ''' Switches the SAS url config attribute

        Easily switch the sasurl configuration variable between
        Utah and local.  Utah sets it to the real API.  Local switches to
        an Ngrok url address

        Parameters:
            sasmode (str):
                the SAS mode to switch to.  Default is Utah

        '''
        assert sasmode in ['utah', 'local'], 'SAS mode can only be utah or local'
        if sasmode == 'local':
            if ngrokid:
                self.sasurl = 'http://{0}.ngrok.io/marvin2/'.format(ngrokid)
            else:
                self.sasurl = 'http://localhost:5000/marvin2/'
        elif sasmode == 'utah':
            self.sasurl = 'https://api.sdss.org/marvin2/'

    def forceDbOff(self):
        ''' Force the database to be turned off '''
        config.db = None
        from marvin import marvindb
        marvindb.forceDbOff()

config = MarvinConfig()
config._checkConfig()

# Inits the Database session and ModelClasses
from marvin.db.marvindb import MarvinDB
marvindb = MarvinDB(dbtype=config.db)

# Inits the URL Route Map
from marvin.api.api import Interaction
config.sasurl = 'https://api.sdss.org/marvin2/'
# config.sasurl = 'http://24147588.ngrok.io/marvin2/'  # this is a temporary measure REMOVE THIS
# config.sasurl = 'http://localhost:5000/marvin2/'
