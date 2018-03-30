
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
    INFO: No release version set. Setting default to MPL-6

    marvin.config.release
    MPL-6

On intial import, Marvin will set the default data version to use the latest MPL available.  You can change the version of MaNGA data using the Marvin :ref:`marvin-config-class`.

.. code-block:: python

    from marvin import config
    config.setRelease('MPL-5')

    config.release
    MPL-5


|

.. _marvin-firststep-cube:

My First Cube
-------------

Now let's play with a Marvin Cube

.. code-block:: python

    # get a cube
    from marvin.tools.cube import Cube
    cc = Cube('8485-1901')

    # we now have a cube object
    print(cc)
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>

    # look at some meta-data
    cc.ra, cc.dec, cc.header['SRVYMODE']
    (232.544703894, 48.6902009334, 'MaNGA dither')

    # look at the quality and target bits
    cc.target_flags
    [<Maskbit 'MANGA_TARGET1' ['SECONDARY_v1_1_0', 'SECONDARY_COM2', 'SECONDARY_v1_2_0']>,
     <Maskbit 'MANGA_TARGET2' []>,
     <Maskbit 'MANGA_TARGET3' []>]

    cc.quality_flag
    <Maskbit 'MANGA_DRP3QUAL' []>

    # get a Spaxel and show its wavelength and flux arrays
    spax = cc[10, 10]

    spax
    <Marvin Spaxel (x=10, y=10)>

    spax.flux.wavelength
    [3621.596, 3622.43, 3623.2642, …,10349.038, 10351.422, 10353.805]A˚[3621.596, 3622.43, 3623.2642, …,10349.038, 10351.422, 10353.805]A˚

    spax.flux
    [0.54676276, 0.46566465, 0.4622981, …,0, 0, 0]1×10−17ergA˚sspaxelcm2[0.54676276, 0.46566465, 0.4622981, …,0, 0, 0]1×10−17ergA˚sspaxelcm2

    # plot the spectrum (you may need matplotlib.pyplot.ion() for interactive display)
    spax.flux.plot()

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

    # list the available map categories and channels (similar to the extensions in a DAP FITS file)
    maps.datamodel

    # get a map using the getMap() method...
    haflux = maps.getMap('emline_gflux', channel='ha_6564')

    # ...or with a shortcut
    haflux2 = maps['emline_gflux_ha_6564']

    # or
    haflux = maps.emline_glflux_ha_6564

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

    # use map arithmetic ( + , - , * , / , or ** )
    niiflux = maps['emline_gflux_nii_6585']
    nii_ha = niiflux / haflux

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
    <ResultSet(set=1/129, index=0:10, count_in_set=10, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', z=0.0171611),
     ResultRow(mangaid=u'1-113403', plate=7815, plateifu=u'7815-12703', ifu_name=u'12703', z=0.0715126),
     ResultRow(mangaid=u'1-113418', plate=7815, plateifu=u'7815-12704', ifu_name=u'12704', z=0.0430806),
     ResultRow(mangaid=u'1-113469', plate=7815, plateifu=u'7815-12702', ifu_name=u'12702', z=0.0394617),
     ResultRow(mangaid=u'1-113520', plate=7815, plateifu=u'7815-1901', ifu_name=u'1901', z=0.0167652),
     ResultRow(mangaid=u'1-113525', plate=8618, plateifu=u'8618-6103', ifu_name=u'6103', z=0.0169457)]

    # NamedTuples can be accessed using dotted syntax (for unique column names) or like normal tuples
    r.results[0].mangaid
    u'1-22438'

    # see the column names
    r.columns
    <ParameterGroup name=Columns, n_parameters=5>
     [<QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=ifu.name, name=ifu_name, short=ifu_name, remote=ifu_name, display=Name>,
     <QueryParameter full=nsa.z, name=z, short=z, remote=z, display=Redshift>]

    # get a list of the full column names
    r.columns.full
    ['cube.mangaid', 'cube.plate', 'cube.plateifu', ifu.name', 'nsa.z']

See the Marvin :ref:`marvin-query` section for more details and examples.  And the :ref:`marvin-query-ref` for the detailed Reference Guide.


No really, go read about :doc:`configuring Marvin <core/config>`, :doc:`Marvin data access modes <core/data-access-modes>`, and :doc:`downloading objects <core/downloads>`.
