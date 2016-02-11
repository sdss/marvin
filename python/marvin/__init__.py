import os


class Config:
    def __init__(self):
        self._mode = 'local'
        self.drpver = None
        self.dapver = None
        self.mplver = None
        self.vermode = None
        self.download = False

        self.sasurl = os.getenv('SAS_URL') if 'SAS_URL' in os.environ else 'https://sas.sdss.org/'

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ['local', 'remote']:
            self._mode = value
        else:
            raise ValueError('config.mode must be "local" or "remote".')

config = Config()
