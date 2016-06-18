
Overview
========

MaNGA
-----

`MaNGA <http://www.sdss.org/surveys/manga/>`_ (Mapping Nearby Galaxies with APO)
is a `SDSS-IV <http://www.sdss.org/>`_ survey to understand the physical
processes that govern the lifecycle of galaxies. MaNGA is obtaining integral
field spectroscopy of 10,000 galaxies at ~kpc spatial scales to probe their
gas-phase and stellar properties as a function of location.  MaNGA produces data
cubes (with two spatial dimensions and one spectral dimension) of the spectra in
individual spatial pixels, called spaxels, and maps of derived properties. This
wealth of spectral information with known spatial inter-connectedness is a
powerful tool for unraveling the mysteries of galaxy evolution, but the massive
scale of the MaNGA survey severely complicates any attempt to harness the full
statistical power of this data set.


Marvin
------

Marvin is a complete ecosystem designed for overcoming the challenge of
searching, accessing, and visualizing the MaNGA data. It consists of a
three-pronged approach of a web app, an importable python package, and an API.
The web app, Marvin-web, provides an easily accesible interface for searching
the MaNGA data and quickly viewing images and spectra of MaNGA galaxies.  The
python package, Marvin-tools, allows users to load data that have on their
computer or retrieve it from the SAS or the database. It provides many
convenience functions and classes for interacting with the data. It also
includes a powerful query functionality that uses the API to query the MaNGA
databases and return the search results to your python session. Marvin-API is
the critical link that allows Marvin-tools and Marvin-web to interact with the
databases, which enables users to harness the statistical power of the MaNGA
data set.


Marvin 1.0
----------

Marvin began as purely a web app

to search on global galaxy properties (sample or DRP),
display images, maps, radial gradients, and a subset of the available spectra
and spectral fits (emline zoom in).

Download data files.


It included comment and tagging functionality to allow for quality assessment of
the MaNGA reduction and analysis pipelines.

Searchable comments


Other
-----
gas-phase (abundances, kinematics)
stellar (kinematics, populations---metallicities, ages)

disruptive technology 

What is Marvin for MaNGA and astronomy community?
What ground-breaking things does it do?
How many spaxels? How many TB of data?
Query on spaxel properties.
Spectral database
DAP database

