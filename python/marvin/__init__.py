import os

class Config:
    def __init__(self):
        self.mode = None
        self.drpver = None
        self.dapver = None
        self.mplver = None
        self.vermode = None
        self.download = False

        self.sasurl = os.getenv('SAS_URL') if 'SAS_URL' in os.environ else 'https://sas.sdss.org/'

config = Config()
