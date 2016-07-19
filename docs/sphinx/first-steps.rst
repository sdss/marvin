
.. _marvin-first-steps:

First Steps
===========

Now that you have installed Marvin, it's time to take your first steps.  If you want to learn more about how Marvin works,
then go see :ref:`marvin-general` to learn about Marvin Modes, Versions, or Downloading.  If you just want to play, then read on.

.. _marvin-firststep:

From your terminal, type ipython.  Ipython is an Interactive Python shell terminal.  It is recommended to always use ipython
instead of python.::

    > ipython

.. jupyter notebook
.. Ctrl-C to exit
.. %matplotlib inline
.. Shift-Enter

Let's import Marvin

.. code-block:: python

    import marvin
    INFO: No MPL or DRP/DAP version set. Setting default to MPL-4

    marvin.config.mplver, marvin.config.drpver, marvin.config.dapver
    MPL-4 v1_5_1 1.1.1

On intial import, Marvin will set the default data version to use as MPL-4.  You can change the version of MaNGA data
using the Marvin :ref:`marvin-config-class`.

.. code-block:: python

    from marvin import config
    config.setMPL('MPL-3')

    config.mplver, config.drpver, config.dapver
    MPL-3 v1_3_3 v1_0_0


|

.. _marvin-firststep-cube:

My First Cube
-------------

Now let's play with a Marvin Cube

.. code-block:: python

    # get a cube
    from marvin.tools.cube import Cube
    cc = Cube(filename='/Users/Brian/Work/Manga/redux/v1_5_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz')

    # we now have a cube object
    print(cc)
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>

    # look at some meta-data
    cc.ra, cc.dec, cc.hdr['SRVYMODE']
    (232.544703894, 48.6902009334, 'MaNGA dither')

    # look at the quality and target bits
    cc.targetbit
    {'bits': [2336L], 'names': ['MNGTRG1']}

    cc.qualitybit
    ('DRP3QUAL', 1L, None)

    # get a Spaxel and show its wavelength and flux arrays
    spax = cc[10,10]

    spax
    <Marvin Spaxel (x=10, y=10)>

    spax.drp.wavelength
    array([  3621.59598486,   3622.42998417,   3623.26417553, ...,
            10349.03843826,  10351.42166679,  10353.80544415])

    spax.drp.flux
    array([-0.00318646,  0.00827731,  0.01482985, ...,  0.        ,
            0.        ,  0.        ], dtype=float32)

    # plot the spectrum (you may need matplotlib.pyplot.ion() for interactive display)
    spax.drp.plot()
    
    # save plot to Downloads directory
    import os
    import matplotlib.pyplot as plt
    plt.savefig(os.getenv('HOME') + '/Downloads/my-first-spectrum.png')

See the Marvin :ref:`marvin-tools` section for more details and examples.  And the :ref:`marvin-tools-ref` for the detailed Reference Guide.

Did you read :ref:`marvin-general` yet?  Do that now!


|

.. _marvin-firststep-query:

My First Query
--------------

Now let's play with a Marvin Query

.. code-block:: python

    # import a Marvin query convenience tool
    from marvin.tools.query import doQuery

    # Do a Query: select all galaxies with NSA redshift < 0.2 and only 19-fiber IFUs
    q, r = doQuery(searchfilter='nsa.z < 0.2 and ifu.name=19*')
    init condition [['nsa.z', '<', '0.2']]
    init condition [['ifu.name', '=', '19*']]
    Your parsed filter is:
    and_(nsa.z<0.2, ifu.name=19*)

    # How many objects met the search criteria?
    r.totalcount
    151

    # Results are returned in chunks of 10 by default
    r.results
    [NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-23023', plate=7992, name=u'1902', z=0.0270670596510172),
     NamedTuple(mangaid=u'1-24099', plate=7991, name=u'1902', z=0.0281657855957747),
     NamedTuple(mangaid=u'1-38103', plate=8082, name=u'1901', z=0.0285587850958109),
     NamedTuple(mangaid=u'1-38157', plate=8083, name=u'1901', z=0.037575539201498),
     NamedTuple(mangaid=u'1-38347', plate=8083, name=u'1902', z=0.036589004099369),
     NamedTuple(mangaid=u'1-43214', plate=8135, name=u'1902', z=0.117997065186501),
     NamedTuple(mangaid=u'1-43629', plate=8143, name=u'1901', z=0.031805731356144),
     NamedTuple(mangaid=u'1-43663', plate=8140, name=u'1902', z=0.0407325178384781),
     NamedTuple(mangaid=u'1-43679', plate=8140, name=u'1901', z=0.0286782365292311)]

    # NamedTuples can be accessed using dotted syntax or like normal tuples
    r.results[0].mangaid
    u'1-22438'

    # see the column names
    r.getColumns()
    [u'mangaid', u'plate', u'name', u'name', u'z']

    # see the full column names
    r.mapColumnsToParams()
    ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']

See the Marvin :ref:`marvin-query` section for more details and examples.  And the :ref:`marvin-query-ref` for the detailed Reference Guide.


No really, go read the :ref:`marvin-general`.
