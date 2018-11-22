.. role:: python(code)
   :language: python

.. _marvin-query-examples:

Example Filter Conditions
=========================

Return Galaxies
---------------

...below a redshift of 0.1::

    nsa.z < 0.1

...on plates 7815 and 8485::

    cube.plate == 7815 or cube.plate == 8485

...with a IFU size of 127::

    ifu.name = 127*

...with g-r color < 0.2::

    g_r < 0.2

...with Halpha flux > 25 in more than 20% of their good spaxels::

    npergood(emline_gflux_ha_6564 > 25) >= 20


Return Spaxels
--------------

...that have an Halpha flux > 25::

    emline_gflux_ha_6564 > 25

...that have Halpha EW > 6 in galaxies between stellar mass 9.5-11 with sersic index < 2::

    nsa.sersic_logmass >= 9.5 and nsa.sersic_logmass < 11 and nsa.sersic_n < 2 and emline_sew_ha_6564 > 6

...that have NII/Halpha ratio > 0.1::

    nii_to_ha > 0.1


Benchmark Queries
-----------------

1. ``emline_gflux_ha_6564 > 25``
2. ``npergood(spaxelprop.emline_gflux_ha_6564 > 5) >= 20``
3. ``nsa.sersic_logmass >= 9.5 and nsa.sersic_logmass < 11 and nsa.sersic_n < 2 and emline_sew_ha_6564 > 6``
4. ``nsa.z < 0.1 and haflux > 25``

|

Benchmark Remote Query Timing in Marvin
---------------------------------------

Query timing exercise post postgres 9.3 config optimization. From a laptop in Baltimore to Utah with explicit queries in remote mode

=====  ===========  =========== ================== ====================
Query  1st Run (s)  2nd Run (s) Best of 5 (1 loop) Best of 5 (10 loops)
=====  ===========  =========== ================== ====================
Q1     0.3012       0.2375      682 ms per loop    710 ms per loop
Q2     fail         fail
Q3     39.493       10.087      7 s per loop       7.19 s per loop
Q4     1.1662       0.3331      783 ms per loop    810 ms per loop
=====  ===========  =========== ================== ====================

::

    q = Query(search_filter=p, mode='remote')

    # Best of 5 (1 loop)
    %timeit -n 1 -r 5 r = q.run()

    # Best of 5 (10 loops)
    %timeit -n 10 -r 5 r = q.run()

|

Benchmark Query Timing Using Raw SQL
------------------------------------

Query timing from pasting the raw SQL into psql at Utah, pre postgres 9.6 config optimization.

=====  ===========  =========== ==================
Query  1st Run (s)  2nd Run (s) Best of 5 (1 loop)
=====  ===========  =========== ==================
Q1     16.994       0.9450      679 ms per loop
Q2     285.67       82.263      30.9 s per loop
Q3     234.72       35.032      33.7 s per loop
Q4     0.7135       0.6954      716 ms per loop
=====  ===========  =========== ==================

Query timing from pasting the raw SQL into psql at Utah, post postgres 9.6 config optimization + parallel queries

=====  ===========  =========== ==================
Query  1st Run (s)  2nd Run (s) Best of 5 (1 loop)
=====  ===========  =========== ==================
Q1     1.0          0.517       514 ms per loop
Q2     31.123       11.120      8.56 s per loop
Q3     81.319       25.612      25.4 s per loop
Q4     0.588        0.5668      566 ms per loop
=====  ===========  =========== ==================
