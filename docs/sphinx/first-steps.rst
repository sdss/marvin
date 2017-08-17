
.. _marvin-first-steps:

First Steps
===========

Now that you have installed Marvin, it's time to take your first steps.

*Side Note*: If you are **new to Python**, we recommend that you check out the `AstroBetter Python <http://www.astrobetter.com/wiki/python>`_ page and `Practical Python for Astronomers <http://python4astronomers.github.io/>`_.  You can also find many general Python tutorials on Google or try typing any specific question you have into Google. If Google returns a link to a similar question asked on `Stack Overflow <http://stackoverflow.com/>`_, then definitely start there.

At any time, you can learn more about :doc:`Marvin configuration settings <core/config>`, :doc:`Marvin data access modes <core/data-access-modes>`, and :doc:`downloading Marvin data <core/downloads>`, but if you just want to play, then read on.

.. _marvin-firststep:

From your terminal, type ``ipython --matplotlib``.  Ipython is an Interactive Python shell terminal, and the ``--matplotlib`` option enables matplotlib support, such as interactive plotting.  It is recommended to always use ipython instead of python.::

    > ipython --matplotlib

.. jupyter notebook
.. Ctrl-C to exit
.. %matplotlib inline
.. Shift-Enter

Let's import Marvin

.. code-block:: python

    import marvin
    INFO: No release version set. Setting default to MPL-5

    marvin.config.release
    MPL-5

On intial import, Marvin will set the default data version to use the latest MPL available.  You can change the version of MaNGA data using the Marvin :ref:`marvin-config-class`.

.. code-block:: python

    from marvin import config
    config.setRelease('MPL-4')

    config.release
    MPL-4


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
    cc.ra, cc.dec, cc.header['SRVYMODE']
    (232.544703894, 48.6902009334, 'MaNGA dither')

    # look at the quality and target bits
    cc.targetbit
    {'bits': [2336L], 'names': ['MNGTRG1']}

    cc.qualitybit
    ('DRP3QUAL', 1L, None)

    # get a Spaxel and show its wavelength and flux arrays
    spax = cc[10, 10]

    spax
    <Marvin Spaxel (x=10, y=10)>

    spax.spectrum.wavelength
    array([  3621.59598486,   3622.42998417,   3623.26417553, ...,
            10349.03843826,  10351.42166679,  10353.80544415])

    spax.spectrum.flux
    array([-0.00318646,  0.00827731,  0.01482985, ...,  0.        ,
            0.        ,  0.        ], dtype=float32)

    # plot the spectrum (you may need matplotlib.pyplot.ion() for interactive display)
    spax.spectrum.plot()

    # save plot to Downloads directory
    import os
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(os.path.expanduser('~'), 'Downloads', 'my-first-spectrum.png'))

See the Marvin :ref:`marvin-tools` section for more details and examples.  And the :ref:`marvin-tools-ref` for the detailed Reference Guide.

Did you read about :doc:`configuring Marvin <core/config>`, :doc:`Marvin data access modes <core/data-access-modes>`, and :doc:`downloading objects <core/downloads>` yet?  Do that now!


|

.. _marvin-firststep-map:

My First Map
------------


.. code-block:: python

    # get a Maps object
    from marvin.tools.maps import Maps
    maps = Maps(mangaid='1-209232')

    print(maps)
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype=SPX, template_kin=GAU-MILESHC)>

    # get the NASA-Sloan Atlas info about the galaxy
    maps.nsa

    # list the available map categories (similar to the extensions in a DAP FITS file)
    maps.properties

    # show the available channels for a map category
    maps.properties['emline_gflux'].channels

    # get a map using the getMap() method...
    haflux = maps.getMap('emline_gflux', channel='ha_6564')

    # ...or with a shortcut
    haflux2 = maps['emline_gflux_ha_6564']

    # If a map category has channels, then specify an individual map by joining the category name
    # (e.g., 'emline_gflux') and channel name (e.g., 'ha_6564') with an underscore
    # (e.g., 'emline_gflux_ha_6564'). Otherwise, just use the category name (e.g., 'stellar_vel').

    # get the map values, inverse variances, and masks
    haflux.value

    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]])

    haflux.ivar
    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]])

    haflux.mask
    array([[1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843, 1073741843],
           [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843, 1073741843],
           [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843, 1073741843],
           ...,
           [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843, 1073741843],
           [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843, 1073741843],
           [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843, 1073741843]])

    # plot the map
    fig, ax = haflux.plot()

    # save plot to Downloads directory
    import os
    fig.savefig(os.path.join(os.path.expanduser('~'), 'Downloads', 'my-first-map.png'))

    # get the central spaxel with getSpaxel()...
    spax = maps.getSpaxel(x=0, y=0)

    # ...or with a shortcut (defaults to xyorig=lower, whereas getSpaxel() defaults to xyorig='center')
    spax2 = maps[17, 17]

    # show the DAP properties
    spax.properties


For more info about maps and the DAP products, check out the DAP Getting Started pages for `MPL-4 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-4/dap/GettingStarted>`_ and `MPL-5 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/dap/GettingStarted>`_.


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

    # How long did my query take?
    r.query_time
    datetime.timedelta(0, 0, 204274)  # a Python datetime timedelta object (days, seconds, microseconds)
    # see total seconds
    r.query_time.total_seconds()
    0.204274

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

    # NamedTuples can be accessed using dotted syntax (for unique column names) or like normal tuples
    r.results[0].mangaid
    u'1-22438'

    # see the column names
    r.getColumns()
    [u'mangaid', u'plate', u'name', u'name', u'z']

    # see the full column names
    r.mapColumnsToParams()
    ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']

See the Marvin :ref:`marvin-query` section for more details and examples.  And the :ref:`marvin-query-ref` for the detailed Reference Guide.


No really, go read about :doc:`configuring Marvin <core/config>`, :doc:`Marvin data access modes <core/data-access-modes>`, and :doc:`downloading objects <core/downloads>`.
