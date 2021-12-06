Marvin
======

Marvin is the ultimate tool to visualise and analyse MaNGA data. It is
developed and maintained by the MaNGA team.

|Build Package| |Coverage Status| |PyPI| |DOI| |astropy|
|readthedocs|

Installation
------------

To painlessly install Marvin:

::

    pip install sdss-marvin

If you would like to contribute to Marvin's development, you can clone
this git repo, and run pip install in editable mode which will install all the
dev dependencies:

::

    git clone https://github.com/sdss/marvin
    cd marvin
    pip install -e .[dev]

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

    @ARTICLE{2019AJ....158...74C,
           author = {{Cherinka}, Brian and {Andrews}, Brett H. and
             {S{\'a}nchez-Gallego}, Jos{\'e} and {Brownstein}, Joel and
             {Argudo-Fern{\'a}ndez}, Mar{\'\i}a and {Blanton}, Michael and
             {Bundy}, Kevin and {Jones}, Amy and {Masters}, Karen and
             {Law}, David R. and {Rowlands}, Kate and {Weijmans}, Anne-Marie and
             {Westfall}, Kyle and {Yan}, Renbin},
            title = "{Marvin: A Tool Kit for Streamlined Access and Visualization of the SDSS-IV MaNGA Data Set}",
          journal = {\aj},
         keywords = {astronomical databases: miscellaneous, methods: data analysis, surveys, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
             year = 2019,
            month = aug,
           volume = {158},
           number = {2},
              eid = {74},
            pages = {74},
              doi = {10.3847/1538-3881/ab2634},
    archivePrefix = {arXiv},
           eprint = {1812.03833},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....158...74C},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }



License
-------

Marvin is licensed under a 3-clause BSD style license - see the
``LICENSE.md`` file.

.. |Build Package| image:: https://github.com/sdss/marvin/actions/workflows/build.yml/badge.svg
   :target: https://github.com/sdss/marvin/actions/workflows/build.yml
.. |Coverage Status| image:: https://coveralls.io/repos/github/sdss/marvin/badge.svg?branch=master
   :target: https://coveralls.io/github/sdss/marvin?branch=master
.. |PyPI| image:: https://img.shields.io/pypi/v/sdss-marvin.svg
   :target: https://pypi.python.org/pypi/sdss-marvin
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596700.svg
   :target: https://doi.org/10.5281/zenodo.596700
.. |astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
.. |readthedocs| image:: https://readthedocs.org/projects/docs/badge/
   :target: http://sdss-marvin.readthedocs.io/en/latest/
