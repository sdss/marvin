Marvin
======

Marvin is the ultimate tool to visualise and analyse MaNGA data. It is
developed and maintained by the MaNGA team.

|Build Status| |Coverage Status| |PyPI| |DOI| |astropy|
|BrowserStack Status| |readthedocs|

Installation
------------

To painlessly install Marvin:

::

    pip install sdss-marvin

If you would like to contribute to Marvin's development, you can clone
this git repo, pip install the dependencies, update the submodules, and
then setup with ``python setup.py install``:

::

    git clone https://github.com/sdss/marvin
    cd marvin
    git submodule update --init --recursive
    pip install -r requirements_dev.txt
    python setup.py develop

What is Marvin?
---------------

Marvin is a complete ecosystem designed for overcoming the challenge of
searching, accessing, and visualizing the MaNGA data. It consists of a
three-pronged approach of a web app, a python package, and an API. The
web app, Marvin-web, provides an easily accessible interface for
searching the MaNGA data and visual exploration of individual MaNGA
galaxies or of the entire sample. The python package, in particular
Marvin-tools, allows users to easily and efficiently interact with the
MaNGA data via local files, files retrieved from the `Science Archive
Server <https://sas.sdss.org>`__, or data directly grabbed from the
database. The tools come mainly in the form of convenience functions and
classes for interacting with the data. An additional tool is a powerful
query functionality that uses the API to query the MaNGA databases and
return the search results to your python session. Marvin-API is the
critical link that allows Marvin-tools and Marvin-web to interact with
the databases, which enables users to harness the statistical power of
the MaNGA data set.

Documentation
-------------

You can find the latest Marvin documentation
`here <http://sdss-marvin.readthedocs.io/en/latest/>`__.

Citation and Acknowledgements
-----------------------------

If you use Marvin for work/research presented in a publication (whether
directly, or as a dependency to another package), we ask that you cite
the `Marvin Software <https://zenodo.org/record/292632>`__ (BibTeX). We
provide the following as a standard acknowledgment you can use if there
is not a specific place to cite the DOI:

::

    *This research made use of Marvin, a core Python package and web framework for MaNGA data, developed by Brian Cherinka, José Sánchez-Gallego, Brett Andrews, and Joel Brownstein. (MaNGA Collaboration, 2018).*

Marvin's Bibtex entry to use:

::

      @misc{brian_cherinka_2018_1146705,
        author       = {Brian Cherinka and
                        José Sánchez-Gallego and
                        Brett Andrews and
                        Joel Brownstein},
        title        = {sdss/marvin: Marvin Beta 2.2.0},
        month        = jan,
        year         = 2018,
        doi          = {10.5281/zenodo.1146705},
        url          = {https://doi.org/10.5281/zenodo.1146705}
      }

License
-------

Marvin is licensed under a 3-clause BSD style license - see the
``LICENSE.md`` file.

.. |Build Status| image:: https://travis-ci.org/sdss/marvin.svg?branch=master
   :target: https://travis-ci.org/sdss/marvin
.. |Coverage Status| image:: https://coveralls.io/repos/github/sdss/marvin/badge.svg?branch=master
   :target: https://coveralls.io/github/sdss/marvin?branch=master
.. |PyPI| image:: https://img.shields.io/pypi/v/sdss-marvin.svg
   :target: https://pypi.python.org/pypi/sdss-marvin
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1146705.svg
   :target: https://doi.org/10.5281/zenodo.1146705
.. |astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
.. |BrowserStack Status| image:: https://www.browserstack.com/automate/badge.svg?badge_key=WWgyaGJBbW45aityUVJtYytDcHFydU9EZE9ObVdOVVpkaUxGZkxwbzdHQT0tLUNkcW5Hc3JaRTdqR0l6ajltSUdTRXc9PQ==--21b221b6714b852f8f4215c787ffa6e2812e2ad6
.. |readthedocs| image:: https://readthedocs.org/projects/docs/badge/
   :target: http://sdss-marvin.readthedocs.io/en/latest/
