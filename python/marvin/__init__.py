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

        self._release = None

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
            drpver, __ = self.lookUpVersions(self.release)
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
    def release(self):
        return self._release

    @release.setter
    def release(self, value):
        value = value.upper()
        if value not in self._mpldict:
            raise MarvinError('trying to set an invalid release version. Valid releases are: {0}'
                              .format(', '.join(sorted(self._mpldict.keys()))))
        self._release = value

        drpver = self._mpldict[self.release][0]
        self.drpall = self._getDrpAllPath(drpver)

    @property
    def session_id(self):
        return bconfig.session_id

    @session_id.setter
    def session_id(self, value):
        bconfig.session_id = value


#################################################

    @property
    def urlmap(self):
        """Retrieves the URLMap the first time it is needed."""

        if self._urlmap is None or (isinstance(self._urlmap, dict) and len(self._urlmap) == 0):
            try:
                response = Interaction('api/general/getroutemap', request_type='get')
            except Exception as e:
                warnings.warn('Cannot retrieve URLMap. Remote functionality will not work: {0}'.format(e),
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
        mpldict = {'MPL-5': ('v2_0_1', '2.0.2'),
                   'MPL-4': ('v1_5_1', '1.1.1'),
                   'MPL-3': ('v1_3_3', 'v1_0_0'),
                   'MPL-2': ('v1_2_0', None),
                   'MPL-1': ('v1_0_0', None)}  # , 'DR13': ('v1_5_4', None)}
        mplsorted = sorted(mpldict.items(), key=lambda p: p[1][0], reverse=True)
        self._mpldict = OrderedDict(mplsorted)

        # Check the versioning config
        if not self.release:
            topkey = self._mpldict.keys()[0]
            log.info('No release version set. Setting default to {0}'.format(topkey))
            self.release = topkey

    def setRelease(self, version):
        """Set the release version.

        Parameters:
            version (str):
                The MPL/DR version to set, in form of MPL-X or DRXX.

        Example:
            >>> config.setRelease('MPL-4')
            >>> config.setRelease('DR13')

        """

        version = version.upper()
        self.release = version

    def setMPL(self, mplver):
        """As :func:`setRelease` but check that the version is and MPL."""

        mm = re.search('MPL-([0-9])', mplver)
        assert mm is not None, 'MPL version must be of form "MPL-[X]"'

        if mm:
            self.setRelease(mplver)

    def setDR(self, drver):
        """As :func:`setRelease` but check that the version is and MPL."""

        mm = re.search('DR1([3-9])', drver)
        assert mm is not None, 'DR version must be of form "DR[XX]"'

        if mm:
            self.setRelease(drver)

    def lookUpVersions(self, release=None):
        """Retrieve the DRP and DAP versions that make up a release version.

        Parameters:
            release (str or None):
                The release version. If ``None``, uses the currently set
                ``release`` value.

        Returns:
            drpver (str):
                The DRP version according to the input MPL version
            dapver (str):
                The DAP version according to the input MPL version

        """

        release = release or self.release

        try:
            drpver, dapver = self._mpldict[release]
        except KeyError:
            raise MarvinError('MPL/DR version {0} not found in lookup table. '
                              'No associated DRP/DAP versions. '
                              'Should they be added?  Check for typos.'.format(release))

        return drpver, dapver

    def lookUpRelease(self, drpver):
        """Retrieve the release version for a given DRP version

        Parameters:
            drpver (str):
                The DRP version to use
        Returns:
            release (str):
                The release version according to the input DRP version
        """

        # Flip the mpldict
        verdict = {val[0]: key for key, val in self._mpldict.items()}

        try:
            release = verdict[drpver]
        except KeyError:
            raise MarvinError('DRP version {0} not found in lookup table. '
                              'No associated MPL version. Should one be added?  '
                              'Check for typos.'.format(drpver))

        return release

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
