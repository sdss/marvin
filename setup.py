# encoding: utf-8
#
# setup.py
#

from setuptools import setup
from astropy.utils.data import download_file
import os
import shutil

maskbits_path = download_file('https://svn.sdss.org/public/repo/sdss/idlutils/'
                              'trunk/data/sdss/sdssMaskbits.par')
shutil.copy(maskbits_path, os.path.join(os.path.dirname(__file__),
                                        'python/marvin/data/',
                                        'sdssMaskbits.par'))


setup()
