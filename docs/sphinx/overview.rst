
.. _marvin-overview:

Overview
========

MaNGA
-----

`MaNGA <http://www.sdss.org/surveys/manga/>`_ (Mapping Nearby Galaxies with APO)
is a `SDSS-IV <http://www.sdss.org/>`_ survey to understand the physical
processes that govern the lifecycle of galaxies. MaNGA is obtaining integral
field spectroscopy of 10,000 galaxies at ~kpc spatial scales to probe their
gas-phase and stellar properties as a function of location.  MaNGA produces 3-D
data cubes (with two spatial dimensions and one spectral dimension), and 2-D
maps of derived properties. This wealth of spectral information with known
spatial inter-connectedness is a powerful tool for unraveling the mysteries of
galaxy evolution, but the massive scale of the MaNGA survey severely complicates
any attempt to harness the full statistical power of this data set.


Marvin
------

Marvin is a complete ecosystem designed for overcoming the challenge of
searching, accessing, and visualizing the MaNGA data. It consists of a
three-pronged approach of a web app, a python package, and an API. The web app,
Marvin-web, provides an easily accessible interface for searching the MaNGA data
and visual exploration of individual MaNGA galaxies or of the entire sample. The
python package, in particular Marvin-tools, allows users to easily and
efficiently interact with the MaNGA data via local files, files retrieved from
the `Science Archive Server <https://sas.sdss.org>`_, or data directly grabbed
from the database.  The tools come mainly in the form of convenience functions
and classes for interacting with the data. An additional tool is a powerful
query functionality that uses the API to query the MaNGA databases and return
the search results to your python session. Marvin-API is the critical link that
allows Marvin-tools and Marvin-web to interact with the databases, which enables
users to harness the statistical power of the MaNGA data set.


Marvin 1.0
----------

Marvin 1.0 began as a pure web app to access, search, view, and comment on MaNGA
galaxies. It allowed users to query on global galaxy properties from the sample
selection catalog or as determined by the reduction pipeline. Users could
download FITS files of the data cubes or the analysis properties. They could
view galaxy images, maps, radial gradients, and a subset of the available
spectra and spectral fits. Users could then comment on plots or tag galaxies,
which was critical for quality assessment of the MaNGA reduction and analysis
pipelines. The comments and tags were also searchable to allow users to see
assessments made by other team members.

Marvin 1.0 relied on many static aspects, such as downloading FITS files and
pre-made png files for the maps, gradients, and spectra. This allowed for fast
development, but ultimately limited scalability and interactvity. The Marvin 2.0
web app moves towards a dynamic model that utilizes the spectral and analysis
properties databases. Much of the underlying functionality in Marvin-web is
helpful for users writing their own analysis software on their own computers, so
we packaged these utilities as an easy to use python module (Marvin-tools).
