
.. currentmodule:: marvin.tools

.. |getAperture| replace:: `~marvin.tools.mixins.aperture.GetApertureMixIn.getAperture`

.. _marvin-get-aperture:

Using |getAperture|
=====================

.. note:: This feature requires `photutils <http://photutils.readthedocs.io/en/stable/>`_ to be availabe. You can install it by typing ``pip install photutils``.

The |getAperture| method, available in `~cube.Cube`, `~maps.Maps`, and `~modelcube.ModelCube`, allows to easily extract the :ref:`spaxels <marvin-subregion-tools>` contained within a geometrical aperture. Let's see it in action ::

    >>> cube = marvin.tools.Cube('8485-1901')
    >>> aperture = cube.getAperture((15, 15), 3, aperture_type='circular')
    >>> aperture
    <MarvinAperture([[15, 15]], r=3.0)>
    >>> aperture.parent
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>
    >>> spaxels = aperture.getSpaxels(threshold=0.8)
    >>> spaxels
    [<Marvin Spaxel (x=14, y=13, loaded=False),
     <Marvin Spaxel (x=15, y=13, loaded=False),
     <Marvin Spaxel (x=16, y=13, loaded=False),
     <Marvin Spaxel (x=13, y=14, loaded=False),
     <Marvin Spaxel (x=14, y=14, loaded=False),
     <Marvin Spaxel (x=15, y=14, loaded=False),
     <Marvin Spaxel (x=16, y=14, loaded=False),
     <Marvin Spaxel (x=17, y=14, loaded=False),
     <Marvin Spaxel (x=13, y=15, loaded=False),
     <Marvin Spaxel (x=14, y=15, loaded=False),
     <Marvin Spaxel (x=15, y=15, loaded=False),
     <Marvin Spaxel (x=16, y=15, loaded=False),
     <Marvin Spaxel (x=17, y=15, loaded=False),
     <Marvin Spaxel (x=13, y=16, loaded=False),
     <Marvin Spaxel (x=14, y=16, loaded=False),
     <Marvin Spaxel (x=15, y=16, loaded=False),
     <Marvin Spaxel (x=16, y=16, loaded=False),
     <Marvin Spaxel (x=17, y=16, loaded=False),
     <Marvin Spaxel (x=14, y=17, loaded=False),
     <Marvin Spaxel (x=15, y=17, loaded=False),
     <Marvin Spaxel (x=16, y=17, loaded=False)]

What happened here? Let's look at this example line by line. We start by initialising the DRP datacube for ``'8485-1901'`` (the :ref:`data access mode <marvin-dma>`, either file or API, does not make a difference). Then we use |getAperture| to define a circular aperture, centred around spaxel ``(15, 15)`` with a radius of 3 pixels. The resulting object is a `~marvin.tools.mixins.aperture.MarvinAperture` that has the Cube as `~marvin.tools.mixins.aperture.MarvinAperture.parent`. Now we can use the `~marvin.tools.mixins.aperture.MarvinAperture.getSpaxels` method to retrieve all the spaxels within the aperture. Note the ``threshold=0.8`` parameter that allows us to define the fraction of the spaxel that need to be within the aperture for it to be extracted (in this case, only spaxels that overlap at least 80% with the aperture are returned). The spaxels are returned with ``loaded=False`` for a faster response. They can be fully loaded using the `~marvin.tools.spaxel.SpaxelBase.load` method.

Defining apertures
------------------

Under the hood, |getAperture| heavily relies on the framework defined in photutils_, and apertures are defined using the same set of conventions. A new aperture is defined by the following parameters:

- A coordinate pair (either 0-indexed ``(x, y)`` pixel coordinates or ``(RA, Dec)`` on-sky coordinates in degrees) or a list of coordinates. If the latter, multiple apertures will be defined, each one centred on a set of coordinates.
- A list of aperture parameters. For a circular aperture this is simply the radius; for a rectangular aperture the width, height, and rotation; and for an elliptical aperture the semi-axis lengths and the rotation. See the `documentation <marvin.tools.mixins.aperture.GetApertureMixIn.getAperture>` for a detailed description of the units and formats. The parameters and on-sky conventions are the same defined by photutils_.
- An ``aperture_type`` keyword parameter indicating the type of aperture. It can be ``'circular'`` (the default), ``'rectangular'``, or ``'elliptical'``.
- A ``coord_type`` keyword parameter that can be ``'pixel'`` or ``'sky'`` to indicate whether the coordinates and aperture parameters are defined in pixels or on-sky coordinates.

The following examples shows how to define several types of aperture ::

    # A rectangular aperture of width 5 pixels, height of 3 pixels, and rotated 30 degrees
     >>> aperture_rect = cube.getAperture((10, 10), (5, 4, 30), aperture_type='rectangular')
     >>> aperture_rect
     <MarvinAperture([[10, 10]], w=5.0, h=4.0, theta=30.0)>

    # Two elliptical apertures, centred around pixels (5, 5) and (20, 15), with
    # major and minor semi-axes 4 and 3 pixels, respectively, non-rotated.
    >>> aperture_ell = cube.getAperture([(5, 5), (20, 15)] (4, 3, 0), aperture_type='elliptical')
    >>> aperture_ell
    <MarvinAperture([[ 5,  5],
                     [20, 15]], a=4.0, b=3.0, theta=0.0)>


Reference
---------

.. autosummary::

   marvin.tools.mixins.aperture.GetApertureMixIn
   marvin.tools.mixins.aperture.MarvinAperture
