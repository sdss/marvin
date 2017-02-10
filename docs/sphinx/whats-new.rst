What's New in Marvin 2.1 (January 2017)
=======================================

* Marvin is now minimally compliant with Python 3.5+

* `<https://sas.sdss.org/marvin>`_ now points to Marvin 2 (instead of Marvin 1).

* The NSA catalog information is now available via **Cube.nsa** in Marvin Cubes.

* Marvin :ref:`marvin-web` now has a new :ref:`nsa_display` tab with interactive scatter, and box-and-whisker plots.

* Marvin :ref:`marvin-web` has more python tips for working with Marvin :ref:`marvin-tools` objects.

* Marvin now uses Sentry to catch and send errors.

* Marvin :ref:`marvin-maps` now include the ability to make and plot a :ref:`marvin-bpt` diagram.

* Marvin :ref:`marvin-maps` have updated plotting display and now include a new signal-to-noise (snr) attribute on each map.

* Check out the :ref:`visual-guide`.

* Marvin Spaxels now include ``ra`` and ``dec`` as properties.

* Streamlined list of query parameters both in the :ref:`marvin-web` and :ref:`marvin-tools`.  Added new parameter ``ha_to_hb`` ratio.

* Marvin has updated the :ref:`marvin-images` functions for downloading, showing, and locating.

* New **check_marvin** utility to provide some basic system checks with regards to Marvin

* Marvin :ref:`marvin-web` now has a "Provide Feedback" button in the navbar that directly links to a New Issue in Github.

* See `Changelog <https://github.com/sdss/marvin/blob/master/CHANGELOG.md>`_ for more.

What's New in Marvin 2.0 Beta (November 2016)
=============================================

* Brand new painless installation (pip install sdss-marvin)

* New Marvin Tools (Maps, Bin, ModelCube)

* Pickling of Marvin Tools, Queries, and Results (i.e. local save and restore)

* DAP Spaxel ("Zonal") Queries

* Dynamic DAP Map display in the web, with point-and-click spaxel

* For MPL-5+, display of model fits in spectrum view in the web

* Versions simplified from mpl, drp, dap down to release

* API :ref:`marvin-authentication`

What's New in Marvin 2.0 Alpha (June 2016)
==========================================

Marvin 2.0 is a complete overhaul of Marvin 1.0, converting Marvin into a full suite of interaction tools.

Marvin 2.0 introduces two new modes of operations, :doc:`tools` and :doc:`api`, to the Marvin
environment, and introduces an extensive redesign of the `Marvin web app
<https://sas.sdss.org/marvin/>`_.

The major improvements and additions in this release:

* :doc:`../data-access-modes`: a new mode based navigation system that allows you to seamlessly interact with MaNGA data no matter where it is.

* :doc:`../tools`: a python package for accessing and interacting with MaNGA
  data, whether the files are in your computer or they need to be retrieved remotely via the
  API.

* :doc:`../api`: remotely grab the data you are looking for as JSONs to integrate directly into your local scripts

* :doc:`../query`: a tool to harness the full statistical power of the MaNGA
  data set by querying the :ref:`marvin-databases`.

* A completely overhauled :doc:`../web` interface, including:

  * A more powerful :ref:`web-search` with an intuitive pseudo-natural language
    search capability.

  * A simple and clean Plate and Galaxy detail page.

  * Interactive spectrum selection from the galaxy image.

  * An image roulette if you are feeling lucky.





