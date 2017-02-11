# Licensed under a 3-clause BSD style license
"""
Marvin is a package intended to simply the access, exploration, and visualization of
the MaNGA dataset for SDSS-IV.  It provides a suite of Python tools, a web interface,
and a REST-like API, under tools/, web/, and api/, respectively.  Core functionality
of Marvin stems from Marvin's Brain.
"""

import os
import re
import warnings
import sys
import marvin
from collections import OrderedDict
from distutils.version import StrictVersion

# Set the Marvin version
try:
    from marvin.version import get_version
except ImportError as e:
    __version__ = 'dev'
else:
    __version__ = get_version()

# Does this so that the implicit module definitions in extern can happen.
from marvin import extern

from marvin.core.exceptions import MarvinUserWarning, MarvinError
from brain.utils.general.general import getDbMachine
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
warnings.filterwarnings('ignore', '(.)+size changed, may indicate binary incompatibility(.)+')


class MarvinConfig(object):
    ''' Global Marvin Configuration

    The global configuration of Marvin.  Use the config object to globally set options for
    your Marvin session.

    Parameters:
        release (str):
            The release version of the MaNGA data you want to use.  Either MPL or DR.
        download (bool):
            Set to turn on downloading of objects with sdss_access
        use_sentry (bool):
            Set to turn on/off the Sentry error logging.  Default is True.
        add_github_message (bool):
            Set to turn on/off the additional Github Issue message in MarvinErrors. Default is True.
        drpall (str):
            The location to your DRPall file, based on which release you have set.
        mode (str):
            The current mode of Marvin.  Either 'auto', 'remote', or 'local'. Default is 'auto'
        sasurl (str):
            The url of the Marvin API on the Utah Science Archive Server (SAS)
        urlmap (dict):
            A dictionary containing the API routing information used by Marvin
        xyorig (str):
            Globally set the origin point for all your spaxel selections.  Either 'center' or 'lower'.
            Default is 'center'
    '''
    def __init__(self):

        self._drpall = None
        self._inapp = False

        self._urlmap = None
        self._xyorig = None

        self._release = None

        self.vermode = None
        self.download = False
        self.use_sentry = True
        self.add_github_message = True

        self._plantTree()
        self._checkSDSSAccess()
        self._check_manga_dirs()
        self._setDbConfig()
        self._checkConfig()
        self._check_netrc()
        self.setDefaultDrpAll()

    def _checkPaths(self, name):
        ''' Check for the necessary path existence.

            This should only run if someone already has TREE_DIR installed
            but somehow does not have a SAS_BASE_DIR, MANGA_SPECTRO_REDUX, or
            MANGA_SPECTRO_ANALYSIS directory
        '''

        name = name.upper()
        if name not in os.environ:
            if name == 'SAS_BASE_DIR':
                path_dir = os.path.expanduser('~/sas')
            elif name == 'MANGA_SPECTRO_REDUX':
                path_dir = os.path.join(os.path.abspath(os.environ['SAS_BASE_DIR']), 'mangawork/manga/spectro/redux')
            elif name == 'MANGA_SPECTRO_ANALYSIS':
                path_dir = os.path.join(os.path.abspath(os.environ['SAS_BASE_DIR']), 'mangawork/manga/spectro/analysis')

            if not os.path.exists(path_dir):
                warnings.warn('no {0}_DIR found. Creating it in {1}'.format(name, path_dir))
                os.makedirs(path_dir)
            os.environ[name] = path_dir

    def _check_netrc(self):
        """Makes sure there is a valid netrc."""

        netrc_path = os.path.join(os.environ['HOME'], '.netrc')

        if not os.path.exists(netrc_path):
            warnings.warn('cannot find a .netrc file in your HOME directory. '
                          'Remote functionality may not work. Go to '
                          'https://api.sdss.org/doc/manga/marvin/api.html#marvin-authentication '
                          'for more information.', MarvinUserWarning)
            return

        if oct(os.stat(netrc_path).st_mode)[-3:] != '600':
            warnings.warn('your .netrc file has not 600 permissions. Please fix it by '
                          'running chmod 600 ~/.netrc. Authentication will not work with '
                          'permissions different from 600.')

    def _check_manga_dirs(self):
        """Check if $SAS_BASE_DIR and MANGA dirs are defined.
           If they are not, creates and defines them.
        """

        self._checkPaths('SAS_BASE_DIR')
        self._checkPaths('MANGA_SPECTRO_REDUX')
        self._checkPaths('MANGA_SPECTRO_ANALYSIS')

    def setDefaultDrpAll(self, drpver=None):
        """Tries to set the default location of drpall.

        Sets the drpall attribute to the location of your DRPall file, based on the
        drpver.  If drpver not set, it is extracted from the release attribute.  It sets the
        location based on the MANGA_SPECTRO_REDUX environment variable

        Parameters:
            drpver (str):
                The DRP version to set.  Defaults to the version corresponding to config.release.
        """

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
                              .format(', '.join(sorted(list(self._mpldict)))))
        self._release = value

        drpver = self._mpldict[self.release][0]
        self.drpall = self._getDrpAllPath(drpver)

    @property
    def session_id(self):
        return bconfig.session_id

    @session_id.setter
    def session_id(self, value):
        bconfig.session_id = value

    @property
    def _traceback(self):
        return bconfig.traceback

    @_traceback.setter
    def _traceback(self, value):
        bconfig.traceback = value

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
                   'MPL-1': ('v1_0_0', None)}  # , 'DR13': ('v1_5_4', None), 'DR14': ('v2_1_1', None)}
        mplsorted = sorted(mpldict.items(), key=lambda p: p[1][0], reverse=True)
        self._mpldict = OrderedDict(mplsorted)

        # Check the versioning config
        if not self.release:
            topkey = list(self._mpldict)[0]
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
            drpver, dapver (tuple):
                A tuple of strings of the DRP and DAP versions according
                to the input MPL version

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

    def switchSasUrl(self, sasmode='utah', ngrokid=None, port=5000, test=False):
        ''' Switches the SAS url config attribute

        Easily switch the sasurl configuration variable between
        utah and local.  utah sets it to the real API.  Local switches to
        use localhost.

        Parameters:
            sasmode ({'utah', 'local'}):
                the SAS mode to switch to.  Default is Utah
            ngrokid (str):
                The ngrok id to use when using a 'localhost' sas mode.
                This assumes localhost server is being broadcast by ngrok
            port (int):
                The port of your localhost server
            test (bool):
                If ``True``, sets the Utah sasurl to the test production, test/marvin2
        '''
        assert sasmode in ['utah', 'local'], 'SAS mode can only be utah or local'
        if sasmode == 'local':
            if ngrokid:
                self.sasurl = 'http://{0}.ngrok.io/marvin2/'.format(ngrokid)
            else:
                self.sasurl = 'http://localhost:{0}/marvin2/'.format(port)
        elif sasmode == 'utah':
            marvin_base = 'test/marvin2/' if test else 'marvin2/'
            self.sasurl = 'https://api.sdss.org/{0}'.format(marvin_base)

    def forceDbOff(self):
        ''' Force the database to be turned off '''
        config.db = None
        from marvin import marvindb
        marvindb.forceDbOff()

    def forceDbOn(self):
        ''' Force the database to be reconnected '''
        self._setDbConfig()
        from marvin import marvindb
        marvindb.forceDbOn(dbtype=self.db)

    def _addExternal(self, name):
        ''' Adds an external product into the path '''
        assert type(name) == str, 'name must be a string'
        externdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extern', name)
        extern_envvar = '{0}_DIR'.format(name.upper())
        os.environ[extern_envvar] = externdir
        pypath = os.path.join(externdir, 'python')
        if os.path.isdir(pypath):
            sys.path.append(pypath)
        else:
            warnings.warn('Python path for external product {0} does not exist'.format(name))

    def _plantTree(self):
        ''' Sets up the sdss tree product root '''
        if 'TREE_DIR' not in os.environ:
            # set up tree using marvin's extern package
            self._addExternal('tree')
            try:
                from tree.tree import Tree
            except ImportError:
                self._tree = None
            else:
                self._tree = Tree(key='MANGA')

    def _checkSDSSAccess(self):
        ''' Checks the client sdss_access setup '''
        if 'SDSS_ACCESS_DIR' not in os.environ:
            # set up sdss_access using marvin's extern package
            self._addExternal('sdss_access')
            try:
                from sdss_access.path import Path
            except ImportError:
                Path = None
            else:
                self._sdss_access_isloaded = True


config = MarvinConfig()

# Inits the Database session and ModelClasses
from marvin.db.marvindb import MarvinDB
marvindb = MarvinDB(dbtype=config.db)

# Inits the URL Route Map
from marvin.api.api import Interaction
config.sasurl = 'https://api.sdss.org/marvin2/'
# config.sasurl = 'http://24147588.ngrok.io/marvin2/'  # this is a temporary measure REMOVE THIS
# config.sasurl = 'http://localhost:5000/marvin2/'

