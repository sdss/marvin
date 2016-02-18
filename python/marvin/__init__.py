import os
import re
import warnings
from marvin.tools.core.exceptions import MarvinUserWarning

# Inits the log
from marvin.tools.core.logger import initLog
log = initLog()

warnings.simplefilter('once')


class Config(object):
    def __init__(self):

        self._mode = 'auto'
        self._drpall = None

        self.drpver = None
        self.dapver = None
        self.mplver = None
        self.vermode = None
        self.download = False
        self.sasurl = os.getenv('SAS_URL') if 'SAS_URL' in os.environ else 'https://sas.sdss.org/'
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

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ['local', 'remote', 'auto']:
            self._mode = value
        else:
            raise ValueError('config.mode must be "local" or "remote".')

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
        # Get machine
        try:
            machine = os.environ['HOSTNAME']
        except:
            machine = None

        # Check if localhost or not
        try:
            localhost = bool(os.environ['MANGA_LOCALHOST'])
        except:
            localhost = machine == 'manga'

        # Check if Utah or not
        try:
            utah = os.environ['UUFSCELL'] == 'kingspeak.peaks'
        except:
            utah = None

        # Check if sas-vm or not
        try:
            sasvm = 'sas-vm' in os.environ['HOSTNAME']
        except:
            sasvm = None

        # Set the dbconfig variable
        if localhost:
            self.db = 'local'
        elif utah or sasvm:
            self.db = 'utah'
        else:
            self.db = None

    def _checkConfig(self):
        ''' Check the config '''
        if not self.mplver or not (self.drpver and self.dapver):
            warnings.warn('No MPL or DRP/DAP version set. Setting default to MPL-4', MarvinUserWarning)
            self.setMPL('MPL-4')

    def setMPL(self, mplver):
        ''' Set the data version by MPL '''

        from marvin.utils.general import lookUpVersions

        m = re.search('MPL-([0-9])', mplver)
        assert m is not None, 'MPL version must be of form "MPL-[X]"'
        if m:
            self.mplver = mplver
            self.drpver, self.dapver = lookUpVersions(mplver)

    def setVersions(self, drpver=None, dapver=None):
        ''' Set the data version by DRPVER and DAPVER '''

        from marvin.utils.general import lookUpMpl

        if drpver:
            assert type(drpver) == str, 'drpver needs to be a string'
            drpre = re.search('v([0-9][_]([0-9])[_]([0-9]))', drpver)
            assert drpre is not None, 'DRP version must be of form "v[X]_[X]_[X]"'
            if drpre:
                self.drpver = drpver
                self.mplver = lookUpMpl(drpver)

        if dapver:
            assert type(dapver) == str, 'dapver needs to be a string'
            dapre1 = re.search('v([0-9][_]([0-9])[_]([0-9]))', dapver)
            dapre2 = re.search('([0-9][.]([0-9])[.]([0-9]))', dapver)
            assert (dapre1 or dapre2) is not None, 'DAP version must be of form "v[X]_[X]_[X]", or [X].[X].[X]'
            if any([dapre1, dapre2]):
                self.dapver = dapver

config = Config()

# Inits the Database session and ModelClasses
session = None
datadb = None
if config.db:
    try:
        from marvin.db.database import db
    except RuntimeError as e:
        print('RuntimeError raised: Problem importing db: {0}'.format(e))
    else:
        try:
            import sdss.internal.database.utah.mangadb.DataModelClasses as datadb
        except Exception as e:
            print('Exception raised: Problem importing mangadb DataModelClasses: {0}'.format(e))
        else:
            session = db.Session()
            datadb = datadb

# Inits the URL Route Map
from marvin.api.api import Interaction
config.sasurl = 'http://5aafb8e.ngrok.com'  # this is a temporary measure REMOVE THIS
response = Interaction('api/general/getroutemap', request_type='get')
config.urlmap = response.getRouteMap()


