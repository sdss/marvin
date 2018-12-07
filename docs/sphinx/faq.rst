
.. _marvin-faq:

Frequently Asked Questions
==========================

General Access
--------------

How do I update marvin?
^^^^^^^^^^^^^^^^^^^^^^^

Just do ``pip install --upgrade sdss-marvin``. Marvin will get updated to the latest
version, along with all the dependencies. If you want to update marvin but keep other
packages in their currrent versions, do
``pip install --upgrade --upgrade-strategy only-if-needed sdss-marvin``. This will only
update dependencies if marvin does need it.

How do report a bug/request a feature?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As Marvin is on Github, we recommend submitting a `Github Issue <https://github.com/sdss/marvin/issues/new>`_.


Why does galaxy X have a thumbnail that is all black?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some galaxies in MaNGA are special targets from one of our `ancillary science programs <http://www.sdss.org/dr14/manga/manga-target-selection/ancillary-targets/>`_.  These targets might end up being outside the main SDSS footprint.  In these cases, the optical image retrieved
using the `Image Cutout Service <http://skyserver.sdss.org/public/en/help/docs/api.aspx#imgcutout>`_ will return a blank black field.


SDSS Collaboration Access
-------------------------

How come I cannot access collaboration data via the API?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please see :ref:`SDSS Collaboration Access<sdss-collaboration-access>`.


How do I login?
^^^^^^^^^^^^^^^
Please see :ref:`Authentication <api-token-auth>`.


Do I have to login in every new python session?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
No! Please see :ref:`Automatically Logging In <auto-login>` for how to set up auto-login.
