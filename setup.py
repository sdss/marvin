# encoding: utf-8
#
# setup.py
#

from setuptools import setup
import os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


filedata = urlopen(
    'https://svn.sdss.org/public/repo/sdss/idlutils/trunk/data/sdss/sdssMaskbits.par')
datatowrite = filedata.read()
with open(os.path.join(os.path.dirname(__file__),
          'python/marvin/data/sdssMaskbits.par'), 'wb') as f:
    f.write(datatowrite)


setup()
