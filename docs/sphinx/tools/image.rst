.. _marvin-image:

Image utilities
===============

:ref:`marvin-tools-image` is a class to interact with MaNGA optical images for each galaxy.


Getting Started
---------------

The ``Image`` tool behaves similar to all other Marvin Tools with regards to multi-modal access.  Call an image with an input and it will either grab it locally or remotely with the API.  It accepts all the standard keywords as the Core Marvin Tools.
::

    from marvin.tools.image import Image

    im = Image('8485-1901')
    print(im)
    <Marvin Image (plateifu='8485-1901', mode='local', data-origin='file')>

    # or remotely
    im = Image('8485-1902')
    MarvinUserWarning: local mode failed. Trying remote now.
    <Marvin Image (plateifu='8485-1902', mode='remote', data-origin='api')>

Marvin will access the image file, and open it using as a PIL Image Object (using the `Python Image Library <http://pillow.readthedocs.io/en/3.1.x/index.html>`_ python package.).  To quickly show the image, use ``Image.show``.
::

    # show the image
    im.show()

The PIL image object is stored in the ``Image.data`` attribute.  You can quickly access a Marvin Cube or Maps associated with this image with the ``getXXX`` methods.
::

    # get a cube
    cube = im.getCube()

    # get a maps
    maps = im.getMaps()

Or conversely, quickly access the image from any of the Marvin Tools
::

    # get the Image from a Cube
    cube = Cube('8485-1901')
    im = cube.getImage()
    print(im)
    <Marvin Image (plateifu='8485-1901', mode='local', data-origin='file')>


.. _marvin-image-using:

Using Image
-----------

Basic Information
^^^^^^^^^^^^^^^^^

Each Image comes with a simple header, ``WCS`` transformation, and the central ``RA, Dec`` coordinate.
::

    # Get the central RA, Dec
    im.ra, im.dec
    (235.57977, 48.465725)

    # Get the WCS information
    im.wcs
    WCS Keywords

    Number of WCS axes: 2
    CTYPE : 'RA---TAN'  'DEC--TAN'
    CRVAL : 235.57977  48.465725
    CRPIX : 281.0  281.0
    PC1_1 PC1_2  : -2.47222222222e-05  0.0
    PC2_1 PC2_2  : 0.0  2.47222222222e-05
    CDELT : 1.0  1.0
    NAXIS : 0  0

Each Image also has a ``bundle`` associated with it.  This is a new utility class, :class:`marvin.utils.general.bundle.Bundle` which provides IFU, sky fiber, and hex coordinates, among other things.
::

    # get the image fiber bundle
    im.bundle
    <Bundle (ra=235.57977, dec=48.465725, ifu=19)>

    # get at the RA, Dec coordinates for the fibers in this bundle
    im.bundle.fibers

Displaying
^^^^^^^^^^

While ``Image.show`` produces a raw image, you have finer control over the image as a Matplotlib figure, using ``Image.plot``.  This renders the image using ``matplotlib.pyplot.imshow``.

.. plot::
    :align: center
    :include-source: True

    # plot the image and return the axis object
    from marvin.tools.image import Image
    im = Image('8553-9102')
    ax = im.plot()

Once you have the plot, you can overlay additional features, such as the IFU or sky fibers, or change the hexagon.

.. plot::
    :align: center
    :include-source: True

    from marvin.tools.image import Image
    im = Image('8553-9102')
    ax = im.plot()

    # overlay the IFU fibers
    im.overlay_fibers(ax)

    # change the style of the hexagon
    im.overlay_hexagon(ax, color='cyan', linewidth=1)

By default the sky fibers are not loaded in the bundle. ``Image.bundle.skies`` will be None.  Overlaying the sky fibers will automatically load them.
::

    # overlay the sky fibers
    im.overlay_skies(ax)

Note however that the sky fibers are often positioned far away from the central galaxy.  If the sky fiber coordinates are outside the range of your image, you will see the error message, ``MarvinError: Cannot overlay sky fibers.  Image is too small.  Please retrieve a bigger image cutout``.  You will need to generate a larger cutout image of the galaxy.

Generating a new Image Cutout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a new utility class, :class:`marvin.utils.general.bundle.Cutout` which provides the ability to generate a new SDSS image cutout using the `SDSS SkyServer Image Cutout Service <http://skyserver.sdss.org/public/en/help/docs/api.aspx#imgcutout>`_.

.. plot::
    :align: center
    :include-source: True

    from marvin.tools.image import Image
    im = Image('8553-9102')

    # generate a new image
    # inputs are height and width in arcsec, and a arcsec/pixel scale
    im.get_new_cutout(100, 100, scale=0.192)

    # plot the new image cutout
    ax = im.plot()

You can also use the Cutout service by itself, in a limited fashion.
::

    from marvin.utils.general import Cutout
    cutout = Cutout(235.57977, 48.465725, 50, 50, scale=0.192)
    cutout.show()

    # save the image
    cutout.save('mycutout.png')

Initializing Lists of Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rather than dealing with individual image objects at a time, you can generate lists of them at once.  You can generate from a straight list of ids.
::

    # generate from a list
    from marvin.tools.image import Image
    images = Image.from_list(['8485-1901', '7443-12701'])

Or by a plateid, getting all the images on the plate
::

    # generate from a plate id
    from marvin.tools.image import Image
    images = Image.by_plate(8485)

Or you can generate a random set of images just for fun
::

    # generate a random list
    from marvin.tools.image import Image
    images = Image.get_random(5)

.. _image-utils:

Utility Functions
^^^^^^^^^^^^^^^^^

The old :ref:`marvin-images` functions documented there have been deprecated and replaced with the following.

* :func:`marvin.utils.general.images.show_image` - replaces **showImage**
* :func:`marvin.utils.general.images.get_images_by_plate` - replaces **getImagesByPlate**
* :func:`marvin.utils.general.images.get_images_by_list` - replaces **getImagesByList**
* :func:`marvin.utils.general.images.get_random_images` - replaces **getRandomImages**

They all work in much the same way except now they utilize the ``Marvin Image`` tool.  The ``get_xxx`` functions now return a list of Marvin Images.  Each function accepts a ``download`` keyword argument that, when set, will download all the images in the list in bulk using ``sdss_access``.


.. _marvin-image-api:

Reference/API
-------------

Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: marvin.tools.image.Image

Class
^^^^^

.. autosummary:: marvin.tools.image.Image

Methods
^^^^^^^

.. autosummary::

    marvin.tools.image.Image.show
    marvin.tools.image.Image.save
    marvin.tools.image.Image.download
    marvin.tools.image.Image.plot
    marvin.tools.image.Image.overlay_hexagon
    marvin.tools.image.Image.overlay_fibers
    marvin.tools.image.Image.overlay_skies
    marvin.tools.image.Image.get_new_cutout
    marvin.tools.image.Image.get_random
    marvin.tools.image.Image.by_plate
    marvin.tools.image.Image.from_list
