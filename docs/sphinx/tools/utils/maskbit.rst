.. _marvin-maskbit:

=====================================================
Maskbit (:mod:`marvin.utils.general.maskbit.Maskbit`)
=====================================================

.. _marvin-utils-maskbit-intro:

Introduction
------------
:mod:`~marvin.utils.general.maskbit.Maskbit` contains the schema for a MaNGA flag (e.g., ``MANGA_DAPPIXMASK``) and provides convenience functions for translating amongst mask values, bits, and labels.


.. _marvin-utils-maskbit-getting-started:

Getting Started
---------------

:mod:`~marvin.utils.general.maskbit.Maskbit` is most frequently encountered as an attribute on a :ref:`marvin-tools` object (e.g., :attr:`marvin.tools.quantities.Map.pixmask` is an instance of :mod:`~marvin.utils.general.maskbit.Maskbit`).

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    ha = maps['gflux ha']

    ha.pixmask
    # <Maskbit 'MANGA_DAPPIXMASK'
    #
    #     bit         label                                        description
    # 0     0         NOCOV                         No coverage in this spaxel
    # 1     1        LOWCOV                        Low coverage in this spaxel
    # 2     2     DEADFIBER                   Major contributing fiber is dead
    # 3     3      FORESTAR                                    Foreground star
    # 4     4       NOVALUE  Spaxel was not fit because it did not meet sel...
    # 5     5    UNRELIABLE  Value is deemed unreliable; see TRM for defini...
    # 6     6     MATHERROR              Mathematical error in computing value
    # 7     7     FITFAILED                  Attempted fit for property failed
    # 8     8     NEARBOUND  Fitted value is too near an imposed boundary; ...
    # 9     9  NOCORRECTION               Appropriate correction not available
    # 10   10     MULTICOMP          Multi-component velocity features present
    # 11   30      DONOTUSE                 Do not use this spaxel for science>


It is also possible to initialize a :mod:`~marvin.utils.general.maskbit.Maskbit` instance without a :ref:`marvin-tools` object:

.. code-block:: python

    from marvin.utils.general.maskbit import Maskbit
    mngtarg1 = Maskbit('MANGA_TARGET1')

    mngtarg1.schema
    #     bit                  label                     description
    # 0     0                   NONE
    # 1     1       PRIMARY_PLUS_COM        March 2014 commissioning
    # 2     2          SECONDARY_COM        March 2014 commissioning
    # 3     3     COLOR_ENHANCED_COM        March 2014 commissioning
    # 4     4         PRIMARY_v1_1_0   First tag, August 2014 plates
    # 5     5       SECONDARY_v1_1_0   First tag, August 2014 plates
    # 6     6  COLOR_ENHANCED_v1_1_0   First tag, August 2014 plates
    # 7     7           PRIMARY_COM2         July 2014 commissioning
    # 8     8         SECONDARY_COM2         July 2014 commissioning
    # 9     9    COLOR_ENHANCED_COM2         July 2014 commissioning
    # 10   10         PRIMARY_v1_2_0
    # 11   11       SECONDARY_v1_2_0
    # 12   12  COLOR_ENHANCED_v1_2_0
    # 13   13                 FILLER                  Filler targets
    # 14   14                RETIRED            Bit retired from use>



.. _marvin-utils-maskbit-using:

Using :mod:`~marvin.utils.general.maskbit.Maskbit`
--------------------------------------------------

Maskbit Schema
``````````````

:mod:`~marvin.utils.general.maskbit.Maskbit` makes properly applying masks easy by providing the schema for a flag:

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    ha = maps['gflux ha']

    ha.manga_target1.description
    # 'Targeting bits for all galaxy targets.'

    ha.manga_target1.schema
    #     bit                  label                     description
    # 0     0                   NONE
    # 1     1       PRIMARY_PLUS_COM        March 2014 commissioning
    # 2     2          SECONDARY_COM        March 2014 commissioning
    # 3     3     COLOR_ENHANCED_COM        March 2014 commissioning
    # 4     4         PRIMARY_v1_1_0   First tag, August 2014 plates
    # 5     5       SECONDARY_v1_1_0   First tag, August 2014 plates
    # 6     6  COLOR_ENHANCED_v1_1_0   First tag, August 2014 plates
    # 7     7           PRIMARY_COM2         July 2014 commissioning
    # 8     8         SECONDARY_COM2         July 2014 commissioning
    # 9     9    COLOR_ENHANCED_COM2         July 2014 commissioning
    # 10   10         PRIMARY_v1_2_0
    # 11   11       SECONDARY_v1_2_0
    # 12   12  COLOR_ENHANCED_v1_2_0
    # 13   13                 FILLER                  Filler targets
    # 14   14                RETIRED            Bit retired from use


Mask, Bits, and Labels
``````````````````````

It also contains the mask value, the corresponding bits, and the corresponding labels for the :ref:`marvin-tools` object:

.. code-block:: python

    ha.manga_target1.mask    # 2336
    ha.manga_target1.bits    # [5, 8, 11]
    ha.manga_target1.labels  # ['SECONDARY_v1_1_0', 'SECONDARY_COM2', 'SECONDARY_v1_2_0']


Array of Mask Values
````````````````````

Let's look at a flag with a mask that is an array and not just a single integer:

.. code-block:: python

    ha.pixmask
    # <Maskbit 'MANGA_DAPPIXMASK'
    #
    #     bit         label                                        description
    # 0     0         NOCOV                         No coverage in this spaxel
    # 1     1        LOWCOV                        Low coverage in this spaxel
    # 2     2     DEADFIBER                   Major contributing fiber is dead
    # 3     3      FORESTAR                                    Foreground star
    # 4     4       NOVALUE  Spaxel was not fit because it did not meet sel...
    # 5     5    UNRELIABLE  Value is deemed unreliable; see TRM for defini...
    # 6     6     MATHERROR              Mathematical error in computing value
    # 7     7     FITFAILED                  Attempted fit for property failed
    # 8     8     NEARBOUND  Fitted value is too near an imposed boundary; ...
    # 9     9  NOCORRECTION               Appropriate correction not available
    # 10   10     MULTICOMP          Multi-component velocity features present
    # 11   30      DONOTUSE                 Do not use this spaxel for science>

    ha.pixmask.mask  # == ha.mask
    # array([[1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843,
    #     1073741843],
    #    ...,
    #    [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843,
    #     1073741843]])

    ha.pixmask.bits
    # [[[0, 1, 4, 30],
    #   ...,
    # [0, 1, 4, 30]]]

    ha.pixmask.labels
    # [[['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE'],
    #   ...,
    # ['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']]]

    ha.pixmask.mask[17, 32]    # 1073741843
    ha.pixmask.bits[17][32]    # [0, 1, 4, 30]
    ha.pixmask.labels[17][32]  # ['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']


Translating Amongst Mask Values, Bits, and labels
`````````````````````````````````````````````````

With ``MANGA_DAPPIXMASK``, you might want to translate individual mask values, bits, or labels:

.. code-block:: python

    ha.pixmask.values_to_bits(1073741843)  # [0, 1, 4, 30]
    ha.pixmask.values_to_labels(1073741843)  #['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']

    # Translate one label
    ha.pixmask.labels_to_value('NOCOV')  # 1
    ha.pixmask.labels_to_bits('NOCOV')   # [0]

    # Translate multiple labels
    ha.pixmask.labels_to_value(['NOCOV', 'UNRELIABLE'])  # 33
    ha.pixmask.labels_to_bits(['NOCOV', 'UNRELIABLE'])  # [0, 5]


Making a Custom Mask
````````````````````

You might want to produce a mask (e.g., to produce a custom mask for plotting):

.. TODO FIX ha.value < 1e-17

.. code-block:: python

    # Mask of regions with no IFU coverage
    nocov = ha.pixmask.get_mask('NOCOV')

    # Mask of regions with low Halpha flux and marked as DONOTUSE
    low_ha = (ha.value < 1e-17) * ha.pixmask.labels_to_value('DONOTUSE')

    # Combine masks using bitwise OR (`|`)
    my_mask = nocov | low_ha

    fig, ax = ha.plot(mask=my_mask)


.. TODO UNCOMMENT
.. .. image:: ../_static/custom_mask.png


See the :ref:`marvin-plotting-tutorial` (e.g., :ref:`marvin-plotting-custom-map-axes`) for more about custom masks.


Reference/API
-------------

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.utils.general.maskbit.Maskbit

.. rubric:: Classes

.. autosummary::

    marvin.utils.general.maskbit.Maskbit

.. rubric:: Functions

.. autosummary::

    marvin.utils.general.maskbit.Maskbit.get_mask
    marvin.utils.general.maskbit.Maskbit.labels_to_bits
    marvin.utils.general.maskbit.Maskbit.labels_to_value
    marvin.utils.general.maskbit.Maskbit.values_to_labels
    marvin.utils.general.maskbit.Maskbit.values_to_bits
