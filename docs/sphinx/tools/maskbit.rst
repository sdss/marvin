.. _marvin-maskbit:

=============================================
Maskbit (:mod:`marvin.tools.maskbit.Maskbit`)
=============================================

.. _marvin-maskbit-intro:

Introduction
------------
:mod:`~marvin.tools.maskbit.Maskbit` is a class for handling maskbits new in 2.2.0.

.. _marvin-maskbit-getting-started:

Getting Started
---------------
To get a map, we first need to create a :mod:`marvin.tools.maps.Maps` object, which contains all of the maps for a galaxy.

.. code-block:: python

    from marvin.tools.maskbit import Maskbit
    maskbit = Maskbit(datamodel='DAP', release='MPL-5')



For more fine-grained data quality control, you can select spaxels based on the :attr:`~marvin.tools.map.Map.mask` attribute, which is an array of DAP spaxel `bitmasks <http://www.sdss.org/dr13/algorithms/bitmasks/>`_ that indicate issues with the data. The following table (lifted from the `MPL-5 Techincal Reference Manual <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/DAPMetaData#MANGA_DAPPIXMASK>`_) gives the meaning of each bit. For MPL-4, the bitmask is simply 0 = good and 1 = bad (which roughly corresponds to DONOTUSE).

===  ============  =============================================================
Bit	 Name	       Description
===  ============  =============================================================
0    NOCOV	       No coverage in this spaxel
1    LOWCOV	       Low coverage in this spaxel
2    DEADFIBER     Major contributing fiber is dead
3    FORESTAR      Foreground star
4    NOVALUE       Spaxel was not fit because it did not meet selection criteria
5    UNRELIABLE    Value is deemed unreliable; see TRM for definition
6    MATHERROR     Mathematical error in computing value
7    FITFAILED     Attempted fit for property failed
8    NEARBOUND     Fitted value is too near an imposed boundary; see TRM
9    NOCORRECTION  Appropriate correction not available
10   MULTICOMP     Multi-component velocity features present
30   DONOTUSE      Do not use this spaxel for science
===  ============  =============================================================

**Note**: For MPL-5, DONOTUSE is a consolidation of the flags NOCOV, LOWCOV, DEADFIBER, FORESTAR, NOVALUE, MATHERROR, FITFAILED, and NEARBOUND.

.. code-block:: python

    import numpy as np
    nocov     = (ha.mask & 2**0) > 0
    lowcov    = (ha.mask & 2**1) > 0
    deadfiber = (ha.mask & 2**2) > 0
    forestar  = (ha.mask & 2**3) > 0
    novalue   = (ha.mask & 2**4) > 0
    matherror = (ha.mask & 2**6) > 0
    fitfailed = (ha.mask & 2**7) > 0
    nearbound = (ha.mask & 2**8) > 0

    bad_data = np.logical_or.reduce((nocov, lowcov, deadfiber, forestar, novalue, matherror, fitfailed, nearbound))
    
    donotuse  = (ha.mask & 2**30) > 0
    
    (bad_data == donotuse).all()  # True


.. _marvin-maskbit-using:

Using :mod:`~marvin.tools.maskbit.Maskbit`
------------------------------------------

Applying Bitmasks to a Map
``````````````````````````

* :doc:`../tutorials/bitmasks`


.. _marvin-maskbit-reference:

Reference/API
-------------

.. rubric:: Class

.. autosummary:: marvin.tools.maskbit.Maskbit

.. rubric:: Methods

.. autosummary::

    .. TODO remove
    marvin.tools.map.Map.save


|