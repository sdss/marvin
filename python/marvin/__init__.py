import os
import re
from marvin.utils.general import lookUpMpl, lookUpVersions


class Config(object):
    def __init__(self):
        self._mode = 'auto'
        self.drpver = None
        self.dapver = None
        self.mplver = None
        self.vermode = None
        self.download = False
        self.sasurl = os.getenv('SAS_URL') if 'SAS_URL' in os.environ else 'https://sas.sdss.org/'
        self._setDbConfig()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ['local', 'remote', 'auto']:
            self._mode = value
        else:
            raise ValueError('config.mode must be "local" or "remote".')

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

    def setMPL(self, mplver):
        ''' Set the data version by MPL '''

        m = re.search('MPL-([0-9])', mplver)
        assert m is not None, 'MPL version must be of form "MPL-[X]"'
        if m:
            self.mplver = mplver
            self.drpver, self.dapver = lookUpVersions(mplver)

    def setVersions(self, drpver=None, dapver=None):
        ''' Set the data version by DRPVER and DAPVER '''

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

# Inits the log
from marvin.tools.core.logger import initLog
log = initLog()
