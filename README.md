# Marvin
Marvin is the ultimate tool to visualise and analyse MaNGA data. It is developed and maintained by the MaNGA team.

[![Build Status](https://travis-ci.org/sdss/marvin.svg?branch=master)](https://travis-ci.org/sdss/marvin)
[![Coverage Status](https://coveralls.io/repos/github/sdss/marvin/badge.svg?branch=master)](https://coveralls.io/github/sdss/marvin?branch=master)
[![PyPI](https://img.shields.io/pypi/v/sdss-marvin.svg)](https://pypi.python.org/pypi/sdss-marvin)
[![PyPI download](https://img.shields.io/pypi/dm/sdss-marvin.svg)](https://pypi.python.org/pypi/sdss-marvin)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.167367.svg)](https://doi.org/10.5281/zenodo.167367)

Installation
------------

To painlessly install Marvin:

    pip install sdss-marvin

If you would like to contribute to Marvin's development, you can clone this git repo, pip install the dependencies, update the submodules, and then setup with `python setup.py install`:

    git clone https://github.com/sdss/marvin
    cd marvin
    pip install -r requirements.txt
    git submodule init
    git submodule update
    python setup.py install


What is Marvin?
---------------

Marvin is a complete ecosystem designed for overcoming the challenge of
searching, accessing, and visualizing the MaNGA data. It consists of a
three-pronged approach of a web app, a python package, and an API. The web app,
Marvin-web, provides an easily accessible interface for searching the MaNGA data
and visual exploration of individual MaNGA galaxies or of the entire sample. The
python package, in particular Marvin-tools, allows users to easily and
efficiently interact with the MaNGA data via local files, files retrieved from
the [Science Archive Server](https://sas.sdss.org), or data directly grabbed
from the database.  The tools come mainly in the form of convenience functions
and classes for interacting with the data. An additional tool is a powerful
query functionality that uses the API to query the MaNGA databases and return
the search results to your python session. Marvin-API is the critical link that
allows Marvin-tools and Marvin-web to interact with the databases, which enables
users to harness the statistical power of the MaNGA data set.

Citation and Acknowledgements
-----------------------------

If you use Marvin for work/research presented in a publication (whether directly, or as a dependency to another package), we ask that you cite the [Marvin Software](https://zenodo.org/record/167367) (BibTeX). We provide the following as a standard acknowledgment you can use if there is not a specific place to cite the DOI:

    *This research made use of Marvin, a core Python package and web framework for MaNGA data, developed by Brian Cherinka,
    José Sánchez-Gallego, and Brett Andrews. (MaNGA Collaboration, 2016).*

Marvin's Bibtex entry to use:

    @misc{brian_cherinka_2016_167367,
      author       = {Brian Cherinka and
                      José Sánchez-Gallego and
                      Brett Andrews},
      title        = {sdss/marvin: Marvin Beta 2.0.7},
      month        = nov,
      year         = 2016,
      doi          = {10.5281/zenodo.167367},
      url          = {https://doi.org/10.5281/zenodo.167367}
    }
