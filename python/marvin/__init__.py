import os

class Config(object):
    def __init__(self):
        self._mode = 'auto'
        self.drpver = None
        self.dapver = None
        self.mplver = None
        self.vermode = None
        self.download = False
        self.session = None
        self.datadb = None
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

config = Config()

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
            config.session = db.Session()
            config.datadb = datadb

# Inits the log
from marvin.tools.core.logger import initLog
log = initLog()
